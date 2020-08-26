/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.helper;


import java.net.*;
import java.io.*;

/**
 *
 * @author technowings
 */
/**
 * This program is a socket client application that connects to a time server to
 * get the current date time.
 *
 * @author www.codejava.net
 */
public class GlottalPythonConnector {

    public static void main(String[] args) {
        checkGlottal("2");
    }

    public static String checkGlottal(String path) {
        String hostname = "localhost";
        int port = 7813;
//        String path = "D:\\work\\project\\PlantDiseaseDetection\\python-cnn\\training_data\\Tomato___Bacterial_spot\\b168.jpg";
        try (Socket socket = new Socket(hostname, port)) {

            OutputStream output = socket.getOutputStream();
            byte[] data = path.getBytes();
            output.write(data);

            InputStream input = socket.getInputStream();
            StringBuffer sb = new StringBuffer();
               data = new byte[1024];
//            while(input.read()!=-1){
            int len = input.read(data);
            if (len != -1) {

//            }C
                System.out.println(new String(data, 0, len));
                return (new String(data, 0, len));

            } else {
                return "unRecognized";
            }



        } catch (UnknownHostException ex) {

            System.out.println("Server not found: " + ex.getMessage());

        } catch (IOException ex) {

            System.out.println("I/O error: " + ex.getMessage());
        }
        return null;
    }
}