#include<iostream>
#include<vector>
#include<cmath>

#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<glm/glm/glm.hpp>
#include<glm/glm/gtc/matrix_transform.hpp>
#include<glm/glm/gtc/type_ptr.hpp>

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
	float a = 2.0f;
	unsigned int indices[] = {
		//上层索引
		0,7,8,
		0,1,8,
		7,6,5,
		7,8,5,
		1,8,3,
		1,2,3,
		8,5,4,
		8,3,4,
		//底层索引
		33,40,41,
		33,34,41,
		40,39,38,
		40,41,38,
		34,41,36,
		34,35,36,
		41,38,37,
		41,36,37,
		//侧面索引
		0,1,10,
		0,9,10,
		1,2,11,
		1,10,11,
		9,10,18,
		9,17,18,
		10,11,19,
		10,18,19,
		17,18,26,
		17,25,26,
		18,19,27,
		18,26,27,
		25,26,34,
		25,33,34,
		26,27,35,
		26,34,35,

		2,3,12,
		2,11,12,
		3,4,13,
		3,12,13,
		11,12,20,
		11,19,20,
		12,13,21,
		12,20,21,
		19,20,28,
		19,27,28,
		20,21,29,
		20,28,29,
		27,28,36,
		27,35,36,
		28,29,37,
		28,36,37,

		4,5,14,
		4,13,14,
		5,6,15,
		5,14,15,
		13,14,22,
		13,21,22,
		14,15,23,
		14,22,23,
		21,22,30,
		21,29,30,
		22,23,31,
		22,30,31,
		29,30,38,
		29,37,38,
		30,31,39,
		30,38,39,

		6,7,16,
		6,15,16,
		7,0,9,
		7,16,9,
		15,16,24,
		15,23,24,
		16,9,17,
		16,24,17,
		23,24,32,
		23,31,32,
		24,17,25,
		24,32,25,
		31,32,40,
		31,39,40,
		32,25,33,
		32,40,33
	};
	//邻接矩阵
	Eigen::MatrixXd adjacency_matrix(42,42);
	adjacency_matrix = Eigen::MatrixXd::Zero(42,42);
	//对角矩阵
	Eigen::MatrixXd diagonal_matrix(42, 42);
	diagonal_matrix = Eigen::MatrixXd::Zero(42, 42);
	for (int i = 0;i < sizeof(indices) / sizeof(unsigned int);i = i+3)
	{
		diagonal_matrix(indices[i], indices[i]) += 0.4;
		diagonal_matrix(indices[i + 1], indices[i + 1]) += 0.4;
		diagonal_matrix(indices[i + 2], indices[i + 2]) += 0.4;

		adjacency_matrix(indices[i], indices[i + 1]) = 1;
		adjacency_matrix(indices[i + 1], indices[i]) = 1;
		adjacency_matrix(indices[i + 1], indices[i + 2]) = 1;
		adjacency_matrix(indices[i + 2], indices[i + 1]) = 1;
	}
	adjacency_matrix = adjacency_matrix / 2;
	//拉普拉斯矩阵
	Eigen::MatrixXd Ls(42, 42); 
	Ls = Ls.setIdentity(42, 42);
	Ls = diagonal_matrix - adjacency_matrix;
	//初始迪卡尔坐标矩阵
	Eigen::MatrixXd coordinate(42, 3);
	coordinate = getCoordinate();
	//拉普拉斯坐标初态
	Eigen::MatrixXd B(42, 3);
	B = Eigen::MatrixXd::Zero(42, 3);
	B = Ls * coordinate;
	//定义锚点
	Eigen::MatrixXd A(12, 42);
	A = Eigen::MatrixXd::Zero(12, 42);
	//固定锚点
	A(0, 33) = 1;
	A(1, 35) = 1;
	A(2, 37) = 1;
	A(3, 39) = 1;
	//移动锚点
	A(4, 0) = 1;
	A(5, 2) = 1;
	A(6, 4) = 1;
	A(7, 6) = 1;
	A(8, 1) = 1;
	A(9, 3) = 1;
	A(10, 5) = 1;
	A(11, 7) = 1;
	//初始化锚点坐标
	Eigen::MatrixXd Ab(12, 3);
	Ab = Eigen::MatrixXd::Zero(12, 3);
	Ab << getAb();
	//转换后迪卡尔坐标
	Eigen::MatrixXd X(54, 3);
	X = Eigen::MatrixXd::Zero(54, 3);
	//拼接后的Ls*矩阵
	Eigen::MatrixXd Ls_star(54,42);
	Ls_star = Eigen::MatrixXd::Zero(54, 42);
	Ls_star << Ls,
				A;

	Eigen::MatrixXd B_star(54, 3);
	B_star = Eigen::MatrixXd::Zero(54, 3);
	B_star << B,
			  Ab;
	Eigen::MatrixXd Laplacian(42, 54);
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
		double vertices[42 * 3] = { 0 };
		for (int i = 0;i < 42;i++)
		{
			for (int j = 0;j < 3;j++)
			{
				vertices[3 * i + j] = X(i, j);
			}
		}
		
		B = Ls * X;

		double r = sqrt(2);
		Ab(4, 0) = r * cos( glfwGetTime() + 135 * PI / 180);
		Ab(4, 1) = r * sin( glfwGetTime() + 135 * PI / 180);
		Ab(5, 0) = r * cos( glfwGetTime() + 225 * PI / 180);
		Ab(5, 1) = r * sin( glfwGetTime() + 225 * PI / 180);
		Ab(6, 0) = r * cos( glfwGetTime() + 315 * PI / 180);
		Ab(6, 1) = r * sin( glfwGetTime() + 315 * PI / 180);
		Ab(7, 0) = r * cos( glfwGetTime() + 45 * PI / 180);
		Ab(7, 1) = r * sin( glfwGetTime() + 45 * PI / 180);
		Ab(8, 0) = cos(glfwGetTime() + 180 * PI / 180);
		Ab(8, 1) = sin(glfwGetTime() + 180 * PI / 180);
		Ab(9, 0) = cos(glfwGetTime() + 270 * PI / 180);
		Ab(9, 1) = sin(glfwGetTime() + 270 * PI / 180);
		Ab(10, 0) = cos(glfwGetTime());
		Ab(10, 1) = sin(glfwGetTime());
		Ab(11, 0) = cos(glfwGetTime() + 90 * PI / 180);
		Ab(11, 1) = sin(glfwGetTime() + 90 * PI / 180);

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
	Eigen::MatrixXd coordinate(42, 3);
	coordinate <<
		-1.0f, 1.0f, 0.0,
		-1.0f, 0.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		0.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f,

		-1.0f, 1.0f, 1.0f,
		-1.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 1.0f,
		0.0f, -1.0f, 1.0f,
		1.0f, -1.0f, 1.0f,
		1.0f, 0.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		0.0f, 1.0f, 1.0f,

		-1.0f, 1.0f, 2.0f,
		-1.0f, 0.0f, 2.0f,
		-1.0f, -1.0f, 2.0f,
		0.0f, -1.0f, 2.0f,
		1.0f, -1.0f, 2.0f,
		1.0f, 0.0f, 2.0f,
		1.0f, 1.0f, 2.0f,
		0.0f, 1.0f, 2.0f,

		-1.0f, 1.0f, 3.0f,
		-1.0f, 0.0f, 3.0f,
		-1.0f, -1.0f, 3.0f,
		0.0f, -1.0f, 3.0f,
		1.0f, -1.0f, 3.0f,
		1.0f, 0.0f, 3.0f,
		1.0f, 1.0f, 3.0f,
		0.0f, 1.0f, 3.0f,

		-1.0f, 1.0f, 4.0f,
		-1.0f, 0.0f, 4.0f,
		-1.0f, -1.0f, 4.0f,
		0.0f, -1.0f, 4.0f,
		1.0f, -1.0f, 4.0f,
		1.0f, 0.0f, 4.0f,
		1.0f, 1.0f, 4.0f,
		0.0f, 1.0f, 4.0f,

		0.0f, 0.0f, 4.0f;
	return coordinate;

}
Eigen::MatrixXd getAb()
{
	Eigen::MatrixXd Ab(12, 3);
	Ab <<
		-1.0, 1.0, 4.0,
		-1.0, -1.0, 4.0,
		1.0, -1.0, 4.0,
		1.0, 1.0, 4.0,
		-1.0, 1.0, 0.0,
		-1.0, -1.0, 0.0,
		1.0, -1.0, 0.0,
		1.0, 1.0, 0.0,
		-1.0, 0.0, 0.0,
		0.0, -1.0, 0.0,
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0;
	return Ab;
}