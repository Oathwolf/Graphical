#include<iostream>
#include<vector>
#include<list>

#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm/glm.hpp>
#include<glm/glm/gtc/matrix_transform.hpp>
#include<glm/glm/gtc/type_ptr.hpp>

//#include"imgui-master/imgui.h"
//#include"imgui-master/imgui_impl_opengl3.h"
//#include"imgui-master/imgui_impl_glfw.h"

#define STB_IMAGE_IMPLEMENTATION
#include<stb-master/stb_image.h>
//#include"Common/camera.h"
#include"Common/readShader.h"

#include"Ray/Random.h"

#include"Ray/RayCamera.h"
#include"Ray/Ray.h"

#include"Ray/Hittable.h"
#include"Ray/RayHittableList.h"
#include"Ray/SphereHit.h"

#include"RayMaterial/RayMaterial.h"

void set_view(GLFWwindow* window, int width, int height);
unsigned int loadTexture(const char* path);

glm::vec3 RayColor(Ray &ray, int depth);

const unsigned int SCR_WIDTH = 600;
const unsigned int SCR_HEIGHT = 600;

float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

RayHittableManager* rayHittableManager = RayHittableManager::GetRayHittableManager();

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "imGui", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glfwSetFramebufferSizeCallback(window, set_view);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	Shader pixelShader("../Shader/pixelShader.vs", "../Shader/pixelShader.fs");
	Shader boradShader("../Shader/boradShader.vs", "../Shader/boradShader.fs");

	float pixelVertex[] = {
		0.0f,0.0f,0.0f
	};
	unsigned int pixelVAO, pixelVBO;
	glGenVertexArrays(1, &pixelVAO);
	glGenBuffers(1, &pixelVBO);
	glBindVertexArray(pixelVAO);
	glBindBuffer(GL_ARRAY_BUFFER, pixelVAO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(pixelVertex), &pixelVertex, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	float boradVertex[] = {
	 // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
	};
	unsigned int boradVAO, boradVBO;
	glGenVertexArrays(1, &boradVAO);
	glGenBuffers(1, &boradVBO);
	glBindVertexArray(boradVAO);
	glBindBuffer(GL_ARRAY_BUFFER, boradVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(boradVertex), &boradVertex, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	unsigned int framebuffer;
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

	unsigned int textureColorBuffer;
	glGenTextures(1, &textureColorBuffer);
	glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer,0);

	unsigned int rbo;
	glGenRenderbuffers(1, &rbo);
	glBindRenderbuffer(GL_RENDERBUFFER, rbo);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, SCR_WIDTH, SCR_HEIGHT);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffr is not complete" << std::endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//Set Ray Camera
	RayCamera rayCamera;
	rayCamera.SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
	rayCamera.SetHorizontal(glm::vec3(1.0f, 0.0f, 0.0f));
	rayCamera.SetHorizontalNum(2.0f);
	rayCamera.SetVertical(glm::vec3(0.0f, 1.0f, 0.0f));
	rayCamera.SetVerticalNum(2.0f);
	rayCamera.SetFocalLength(1.0f);

	glm::vec3 left_down = rayCamera.GetPosition()
		- rayCamera.GetRealHorizontal() / 2.0f
		- rayCamera.GetRealVertical() / 2.0f
		+ glm::vec3(0.0f, 0.0f, rayCamera.GetFocalLength());

	//Set Scene
	std::shared_ptr<Hittable> pHittable;

	pHittable = std::make_shared<StandSphere>();
	rayHittableManager->InputHittableList(pHittable);
	pHittable = std::make_shared<BigSphere>();
	rayHittableManager->InputHittableList(pHittable);
	pHittable = std::make_shared<MirrorSphere>();
	rayHittableManager->InputHittableList(pHittable);
	pHittable = std::make_shared<MirrorDiffuseSphere>();
	rayHittableManager->InputHittableList(pHittable);
	/*pHittable = std::make_shared<GlassSphere>();
	rayHittableManager->InputHittableList(pHittable);*/

	//render ray tracing
	int sampleRayNum = 10;
	glViewport(0, 0, SCR_WIDTH, SCR_HEIGHT);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	pixelShader.use();
	for (int i = 0;i < SCR_HEIGHT;++i)
	{
		for (int j = 0;j < SCR_WIDTH;++j)
		{
			glm::vec3 position = glm::vec3(0.0f, 0.0f, 0.0f);
			position.y = float(2 * i) / float(SCR_HEIGHT) - 1.0f;
			position.x = float(2 * j) / float(SCR_WIDTH) - 1.0f;
			pixelShader.setVec3("position", position);

			glm::vec3 totalColor = glm::vec3(0, 0, 0);
			for (int s = 0;s < sampleRayNum;++s)
			{
				glm::vec3 ray_direction = left_down +
					rayCamera.GetRealHorizontal() / float(SCR_WIDTH)*(float(j) + GetCommonRandom(-1, 1)) +
					rayCamera.GetRealVertical() / float(SCR_HEIGHT)*(float(i) + GetCommonRandom(-1, 1));
				Ray ray;
				ray.SetColor(glm::vec3(1.0, 1.0, 1.0));
				ray.SetOrigin(rayCamera.GetPosition());
				ray.SetDirection(ray_direction);
				totalColor += RayColor(ray,0);
			}
			totalColor = totalColor / (float)sampleRayNum;
			totalColor.x = sqrt(totalColor.x);
			totalColor.y = sqrt(totalColor.y);
			totalColor.z = sqrt(totalColor.z);
			pixelShader.setVec3("color", totalColor);
			glBindVertexArray(pixelVAO);
			glDrawArrays(GL_POINTS, 0, 1);
		}
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	boradShader.use();
	boradShader.setInt("boradTexture", 0);

	while (!glfwWindowShouldClose(window))
	{
		glClearColor(0.1f,0.1f,0.1f,1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		
		boradShader.use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
		glBindVertexArray(boradVAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;

}
glm::vec3 RayColor(Ray &ray, int depth)
{
	if (depth > 5)
	{
		return glm::vec3(0, 0, 0);
	}
	glm::vec3 return_color = glm::vec3(0, 0, 0);
	Ray copy_ray = ray;
	std::shared_ptr<Hittable> pHittable = nullptr;
	for (std::list<std::shared_ptr<Hittable>>::iterator it = 
		rayHittableManager->GetHittableList()->begin();
		it != rayHittableManager->GetHittableList()->end();it++)
	{
		if ((*it)->TestHit(copy_ray))
		{
			pHittable = *it;
		}
	}
	if (pHittable != nullptr)
	{
		pHittable->Hit(ray);
		return_color += RayColor(ray, depth + 1);
		ray.SetColor(return_color);
	}
	else
	{
		glm::vec3 ray_color = ray.GetColor();
		glm::vec3 background_color = glm::vec3(1.0f, 1.0f, 1.0f);
		ray_color.x *= background_color.x;
		ray_color.y *= background_color.y;
		ray_color.z *= background_color.z;
		ray.SetColor(ray_color);
	}
	return ray.GetColor();
}








void set_view(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}
unsigned int loadTexture(char const* path)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	int width, height, nrChannels;
	unsigned char* data = stbi_load(path, &width, &height, &nrChannels, 0);
	if (data)
	{
		GLenum format;
		if (nrChannels == 1)
			format = GL_RED;
		else if (nrChannels==3)
		{
			format = GL_RGB;
		}
		else if(nrChannels ==4)
		{
			format = GL_RGBA;
		}
		glBindTexture(GL_TEXTURE_2D, textureID);
		glTexImage2D(GL_TEXTURE_2D, 0,format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		stbi_image_free(data);
	}
	else
	{
		std::cout << "Texture failed to load at path" << path << std::endl;
		stbi_image_free(data);
	}
	return textureID;
}