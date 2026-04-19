#!/usr/bin/env python3
# 4h_RSI2_With_Trend_Filter
# Hypothesis: 4h RSI(2) extreme reversion with EMA(50) trend filter
# RSI(2) < 10 signals oversold bounce in uptrend (EMA50 up)
# RSI(2) > 90 signals overbought rejection in downtrend (EMA50 down)
# Low-frequency signals (2-5/year) avoid fee drag; trend filter avoids counter-trend traps
# Works in bull via long signals in uptrend, bear via short signals in downtrend

name = "4h_RSI2_With_Trend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(2) - very short period for extreme readings
    def calculate_rsi(close_prices, period=2):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # Initial values
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # EMA(50) for trend filter
    def calculate_ema(close_prices, period=50):
        ema = np.zeros_like(close_prices)
        if len(close_prices) >= 1:
            ema[0] = close_prices[0]
            alpha = 2.0 / (period + 1)
            for i in range(1, len(close_prices)):
                ema[i] = alpha * close_prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    rsi2 = calculate_rsi(close, 2)
    ema50 = calculate_ema(close, 50)
    
    # Trend direction: EMA50 slope
    ema50_slope = np.zeros_like(ema50)
    ema50_slope[1:] = ema50[1:] - ema50[:-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Long setup: RSI2 oversold + uptrend
        if rsi2[i] < 10 and ema50_slope[i] > 0:
            if position <= 0:  # Reverse from short or enter long from flat
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25  # Maintain long
        # Short setup: RSI2 overbought + downtrend
        elif rsi2[i] > 90 and ema50_slope[i] < 0:
            if position >= 0:  # Reverse from long or enter short from flat
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25  # Maintain short
        # Exit conditions: RSI2 returns to neutral zone
        elif 40 <= rsi2[i] <= 60:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals