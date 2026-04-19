#!/usr/bin/env python3

# 4h_RSI_Trend_Volume_Crossover
# Hypothesis: 4h RSI(14) crossing above/below 50 with trend confirmation (EMA21) and volume spike.
# RSI crossing 50 captures momentum shifts; EMA21 filters direction; volume confirms institutional interest.
# Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year).
# Works in bull/bear via EMA21 trend filter - only long when price > EMA21, short when price < EMA21.

name = "4h_RSI_Trend_Volume_Crossover"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        # Wilder smoothing
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # EMA21 for trend
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        alpha = 2.0 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
    
    # 4h data for indicators
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate indicators on 4h data
    rsi_4h = calculate_rsi(df_4h['close'].values, 14)
    ema21_4h = calculate_ema(df_4h['close'].values, 21)
    
    # Align to lower timeframe (though we're using 4h as primary, this ensures proper alignment)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    ema21_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (strict for fewer trades)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema21_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price > EMA21 for long bias, price < EMA21 for short bias
        bullish_trend = close[i] > ema21_aligned[i]
        bearish_trend = close[i] < ema21_aligned[i]
        
        if position == 0:
            # Long: RSI crosses above 50 with bullish trend and volume
            if (rsi_aligned[i] > 50 and 
                rsi_aligned[i-1] <= 50 and  # crossed above
                bullish_trend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 50 with bearish trend and volume
            elif (rsi_aligned[i] < 50 and 
                  rsi_aligned[i-1] >= 50 and  # crossed below
                  bearish_trend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI crosses below 50 or trend turns bearish
            if (rsi_aligned[i] < 50 and rsi_aligned[i-1] >= 50) or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI crosses above 50 or trend turns bullish
            if (rsi_aligned[i] > 50 and rsi_aligned[i-1] <= 50) or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals