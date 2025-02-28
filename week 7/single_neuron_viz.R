library(shiny)
library(ggplot2)

# UI definition
ui <- fluidPage(
  titlePanel("Single Neuron Visualization"),
  
  sidebarLayout(
    sidebarPanel(
      # Input value slider
      sliderInput("input_value", "Input Value (x):",
                  min = -10, max = 10, value = 2, step = 0.1),
      
      # Weight slider
      sliderInput("weight", "Weight (w):",
                  min = -5, max = 5, value = 1, step = 0.1),
      
      # Bias slider
      sliderInput("bias", "Bias (b):",
                  min = -5, max = 5, value = 0, step = 0.1),
      
      # Activation function selector
      selectInput("activation", "Activation Function:",
                  choices = c("Linear", "ReLU", "Sigmoid", "Tanh")),
      
      # Neuron formula
      verbatimTextOutput("formula"),
      
      # Current calculation
      verbatimTextOutput("calculation")
    ),
    
    mainPanel(
      # Neuron diagram
      plotOutput("neuronPlot"),
      
      # Activation function plot
      plotOutput("activationPlot")
    )
  )
)

# Server logic
server <- function(input, output) {
  # Activation functions
  activation_functions <- list(
    Linear = function(x) x,
    ReLU = function(x) pmax(0, x),
    Sigmoid = function(x) 1 / (1 + exp(-x)),
    Tanh = function(x) tanh(x)
  )
  
  # Calculate the neuron output
  output_value <- reactive({
    # Calculate weighted input + bias
    z <- input$input_value * input$weight + input$bias
    
    # Apply the selected activation function
    activation_functions[[input$activation]](z)
  })
  
  # Display the formula
  output$formula <- renderText({
    act_func <- input$activation
    paste0(
      "Neuron Formula:\n",
      "z = x * w + b\n",
      "output = ", act_func, "(z)\n",
      "\nWhere:\n",
      "x = input value\n",
      "w = weight\n",
      "b = bias\n",
      act_func, " = activation function"
    )
  })
  
  # Display the calculation
  output$calculation <- renderText({
    z <- input$input_value * input$weight + input$bias
    out <- output_value()
    
    paste0(
      "Current Calculation:\n",
      "z = ", input$input_value, " * ", input$weight, " + ", input$bias, " = ", round(z, 4), "\n",
      "output = ", input$activation, "(", round(z, 4), ") = ", round(out, 4)
    )
  })
  
  # Render the neuron diagram
  output$neuronPlot <- renderPlot({
    par(mar = c(1, 1, 1, 1))
    plot(0, 0, type = "n", xlim = c(0, 100), ylim = c(0, 100), 
         xaxt = "n", yaxt = "n", xlab = "", ylab = "", bty = "n")
    
    # Draw the neuron
    symbols(30, 50, circles = 15, inches = FALSE, add = TRUE, bg = "lightblue")
    symbols(70, 50, circles = 15, inches = FALSE, add = TRUE, bg = "lightgreen")
    
    # Draw the connection with weight
    arrows(45, 50, 55, 50, lwd = 2, length = 0.1)
    text(50, 55, paste("Weight =", input$weight), cex = 1.2)
    
    # Label input and output
    text(30, 50, "x", cex = 1.5)
    text(70, 50, "output", cex = 1.2)
    
    # Show bias
    text(70, 35, paste("Bias =", input$bias), cex = 1.2)
    
    # Show activation function
    text(70, 65, paste("Activation:", input$activation), cex = 1.2)
    
    # Show current values
    text(30, 80, paste("Input =", input$input_value), cex = 1.2)
    text(70, 80, paste("Output =", round(output_value(), 4)), cex = 1.2)
  })
  
  # Render the activation function plot
  output$activationPlot <- renderPlot({
    x_vals <- seq(-10, 10, length.out = 1000)
    z_vals <- x_vals * input$weight + input$bias
    y_vals <- activation_functions[[input$activation]](z_vals)
    
    df <- data.frame(x = x_vals, y = y_vals)
    
    # Mark the current input-output point
    current_x <- input$input_value
    current_z <- current_x * input$weight + input$bias
    current_y <- activation_functions[[input$activation]](current_z)
    
    # Create the plot
    p <- ggplot(df, aes(x = x, y = y)) +
      geom_line(color = "blue", size = 1) +
      geom_point(aes(x = current_x, y = current_y), color = "red", size = 4) +
      geom_vline(xintercept = 0, linetype = "dashed", alpha = 0.5) +
      geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
      labs(
        title = paste(input$activation, "Activation Function"),
        x = "Input (x)",
        y = "Output"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
    
    # Add appropriate y-axis limits based on activation function
    if (input$activation == "Sigmoid") {
      p <- p + ylim(-0.1, 1.1)
    } else if (input$activation == "Tanh") {
      p <- p + ylim(-1.1, 1.1)
    }
    
    p
  })
}

# Run the app
shinyApp(ui = ui, server = server)