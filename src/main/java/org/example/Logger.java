package org.example;

public class Logger {
    public static void log(String s){
        Main.log.info(s);
        System.out.println(s);
    }
}
