#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Camarilla pivot levels with volume confirmation
# - Uses 1d Camarilla pivot levels (R3, S3, R4, S4) for institutional support/resistance
# - Enters long at S3 bounce with volume confirmation, exits at R3
# - Enters short at R3 rejection with volume confirmation, exits at S3
# - Uses 6h RSI < 40 for long entries and > 60 for short entries to avoid overextended moves
# - Designed to work in both bull and bear markets by fading extremes with institutional levels
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dCamarilla_S3R3_Bounce_Volume_RSI"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_val * 1.1 / 2)
    S3 = pivot - (range_val * 1.1 / 2)
    R4 = pivot + (range_val * 1.1)
    S4 = pivot - (range_val * 1.1)
    
    # Align 1d Camarilla levels to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume filter (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Significant volume confirmation
    
    # RSI filter (6h timeframe) - avoid overextended entries
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_values = calculate_rsi(close, 14)
    rsi_long_filter = rsi_values < 40  # Oversold for long
    rsi_short_filter = rsi_values > 60  # Overbought for short
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(rsi_long_filter[i]) or 
            np.isnan(rsi_short_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bounce at S3 with volume and RSI confirmation
            if (low[i] <= S3_6h[i] * 1.001 and  # Allow small tolerance for wicks
                close[i] > S3_6h[i] and 
                volume_spike[i] and 
                rsi_long_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: rejection at R3 with volume and RSI confirmation
            elif (high[i] >= R3_6h[i] * 0.999 and  # Allow small tolerance for wicks
                  close[i] < R3_6h[i] and 
                  volume_spike[i] and 
                  rsi_short_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: reach R3 or break below S4 (stop)
            if close[i] >= R3_6h[i] * 0.999 or close[i] <= S4_6h[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: reach S3 or break above R4 (stop)
            if close[i] <= S3_6h[i] * 1.001 or close[i] >= R4_6h[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals