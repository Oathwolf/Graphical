
#include<iostream>
#include<vector>
#include<cmath>

#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm/glm.hpp>
#include<glm/glm/gtc/matrix_transform.hpp>
#include<glm/glm/gtc/type_ptr.hpp>

//#include<assimp/Importer.hpp>
//#include<assimp/scene.h>
//#include<assimp/postprocess.h>

#define STB_IMAGE_IMPLEMENTATION
#include<stb-master/stb_image.h>
#include"Common/camera.h"
#include"Common/readShader.h"

#include<eigen/Eigen/Dense>

#define PI 3.1415926

void set_view(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void input(GLFWwindow* window);
unsigned int loadTexture(const char* path);

Eigen::MatrixXd getCoordinate();
Eigen::MatrixXd getAb();

const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 1000;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = (float)SCR_WIDTH / 2.0;
float lastY = (float)SCR_HEIGHT / 2.0;
bool firstMouse = true;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

float deltaTime = 0.0f;
float lastTime = 0.0f;

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_ANY_PROFILE, GLFW_OPENGL_ANY_PROFILE);

	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Laplacian", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, set_view);
	glfwSetCursorPosCallback(window, mouse_callback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}
	glEnable(GL_DEPTH_TEST);
	Shader shader("../Shader/base.vs", "../Shader/base.fs");
	unsigned int index[] = {
		0,1,8,0,7,8
	};
	unsigned int indices[72 * 3] = {0};

	for (int i = 0;i < 6;i++)
	{
		for (int j = 0;j < 6;j++)
		{
			for (int k = 0;k < 6 ;k++)
			{
				indices[k + 6 * j + 36 * i] = index[k] + j + 7*i;
			}
		}
	}
	for (int i = 0;i < 72 * 3;i++)
	{
		printf("%d ", indices[i]);
	}
	//邻接矩阵
	Eigen::MatrixXd adjacency_matrix(49,49);
	adjacency_matrix = Eigen::MatrixXd::Zero(49,49);
	//对角矩阵
	Eigen::MatrixXd diagonal_matrix(49, 49);
	diagonal_matrix = Eigen::MatrixXd::Zero(49, 49);
	for (int i = 0;i < sizeof(indices) / sizeof(unsigned int);i = i+3)
	{
		diagonal_matrix(indices[i], indices[i]) += 0.7;
		diagonal_matrix(indices[i + 1], indices[i + 1]) += 0.7;
		diagonal_matrix(indices[i + 2], indices[i + 2]) += 0.7;

		adjacency_matrix(indices[i], indices[i + 1]) = 1;
		adjacency_matrix(indices[i + 1], indices[i]) = 1;
		adjacency_matrix(indices[i + 1], indices[i + 2]) = 1;
		adjacency_matrix(indices[i + 2], indices[i + 1]) = 1;
	}
	adjacency_matrix = adjacency_matrix / 2;
	//拉普拉斯矩阵
	Eigen::MatrixXd Ls(49, 49); 
	Ls = Ls.setIdentity(49, 49);
	Ls = diagonal_matrix - adjacency_matrix;
	//初始迪卡尔坐标矩阵
	Eigen::MatrixXd coordinate(49, 3);
	coordinate = getCoordinate();
	//拉普拉斯坐标初态
	Eigen::MatrixXd B(49, 3);
	B = Eigen::MatrixXd::Zero(49, 3);
	B = Ls * coordinate;
	//定义锚点
	Eigen::MatrixXd A(1, 49);
	A = Eigen::MatrixXd::Zero(1, 49);
	A(0, 24) = 1;
	//初始化锚点坐标
	Eigen::MatrixXd Ab(1, 3);
	Ab = Eigen::MatrixXd::Zero(1, 3);
	Ab << getAb();
	//转换后迪卡尔坐标
	Eigen::MatrixXd X(50, 3);
	X = Eigen::MatrixXd::Zero(50, 3);
	//拼接后的Ls*矩阵
	Eigen::MatrixXd Ls_star(50,49);
	Ls_star = Eigen::MatrixXd::Zero(50, 49);
	Ls_star << Ls,
				A;

	Eigen::MatrixXd B_star(50, 3);
	B_star = Eigen::MatrixXd::Zero(50, 3);
	B_star << B,
			  Ab;
	Eigen::MatrixXd Laplacian(49, 50);
	Laplacian = (Ls_star.transpose()*Ls_star).inverse()*Ls_star.transpose();

	unsigned int VBO, VAO, EBO;
	glGenVertexArrays(1, &VAO);
	/*glGenBuffers(1, &VBO);*/
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	/*glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_DYNAMIC_DRAW);*/

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), &indices, GL_STATIC_DRAW);

	/*glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);*/
	while (!glfwWindowShouldClose(window))
	{
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastTime;
		lastTime = currentFrame;

		input(window);
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
		shader.use();
		shader.setVec3("objectColor", 1.0f, 0.5f, 0.31f);
		shader.setVec3("lightColor", 1.0f, 1.0f, 1.0f);
		shader.setVec3("lightPos", lightPos);
		glm::mat4 model = glm::mat4(1.0f);
		glm::mat4 view = camera.GetViewMatrix();
		glm::mat4 projection = glm::mat4(1.0f);
		view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
		projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		shader.setMat4("model", model);
		shader.setMat4("view", view);
		shader.setMat4("projection", projection);  

		X = Laplacian*B_star;
		double vertices[49 * 3] = { 0 };
		for (int i = 0;i < 49;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				vertices[3 * i + j] = X(i, j);
			}
		}
		
		B = Ls * X;

		Ab(0, 2) = 5 * sin(glfwGetTime() / 3);

		B_star << B,
			Ab;

		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), &vertices, GL_STREAM_DRAW);
		glVertexAttribPointer(0, 3, GL_DOUBLE, GL_FALSE, 3 * sizeof(double), (void*)0);
		glEnableVertexAttribArray(0);

		glBindVertexArray(VAO);

		glDrawElements(GL_TRIANGLES, sizeof(indices)/sizeof(unsigned int), GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(window);
		glfwPollEvents();  
	}
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	glDeleteVertexArrays(1, &VAO);
	glfwTerminate();
	return 0;
}

void input(GLFWwindow* window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		camera.ProcessKeyboard(FORWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		camera.ProcessKeyboard(BACKWARD, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		camera.ProcessKeyboard(LEFT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		camera.ProcessKeyboard(RIGHT, deltaTime);
	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
void set_view(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos;

	lastX = xpos;
	lastY = ypos;

	camera.ProcessMouseMovement(xoffset,yoffset);
}
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	camera.ProcessMouseScroll(yoffset);
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
Eigen::MatrixXd getCoordinate()
{
	Eigen::MatrixXd coordinate(49, 3);
	coordinate <<
		-3.0, 3.0, 0.0,
		-2.0, 3.0, 0.0,
		-1.0, 3.0, 0.0,
		0.0, 3.0, 0.0,
		1.0, 3.0, 0.0,
		2.0, 3.0, 0.0,
		3.0, 3.0, 0.0,
		-3.0, 2.0, 0.0,
		-2.0, 2.0, 0.0,
		-1.0, 2.0, 0.0,
		0.0, 2.0, 0.0,
		1.0, 2.0, 0.0,
		2.0, 2.0, 0.0,
		3.0, 2.0, 0.0,
		-3.0, 1.0, 0.0,
		-2.0, 1.0, 0.0,
		-1.0, 1.0, 0.0,
		0.0, 1.0, 0.0,
		1.0, 1.0, 0.0,
		2.0, 1.0, 0.0,
		3.0, 1.0, 0.0,
		-3.0, 0.0, 0.0,
		-2.0, 0.0, 0.0,
		-1.0, 0.0, 0.0,
		0.0, 0.0, 0.0,
		1.0, 0.0, 0.0,
		2.0, 0.0, 0.0,
		3.0, 0.0, 0.0,
		-3.0, -1.0, 0.0,
		-2.0, -1.0, 0.0,
		-1.0, -1.0, 0.0,
		0.0, -1.0, 0.0,
		1.0, -1.0, 0.0,
		2.0, -1.0, 0.0,
		3.0, -1.0, 0.0,
		-3.0, -2.0, 0.0,
		-2.0, -2.0, 0.0,
		-1.0, -2.0, 0.0,
		0.0, -2.0, 0.0,
		1.0, -2.0, 0.0,
		2.0, -2.0, 0.0,
		3.0, -2.0, 0.0,
		-3.0, -3.0, 0.0,
		-2.0, -3.0, 0.0,
		-1.0, -3.0, 0.0,
		0.0, -3.0, 0.0,
		1.0, -3.0, 0.0,
		2.0, -3.0, 0.0,
		3.0, -3.0, 0.0;
	return coordinate;

}
Eigen::MatrixXd getAb()
{
	Eigen::MatrixXd Ab(1, 3);
	Ab <<
		0.0, 0.0, 0.0;
	return Ab;
}