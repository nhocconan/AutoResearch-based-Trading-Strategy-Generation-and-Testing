#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe using 1-day and 1-week RSI with mean reversion strategy.
# Long when 1-day RSI < 30 AND 1-week RSI < 40, short when 1-day RSI > 70 AND 1-week RSI > 60.
# Uses volume confirmation (volume > 1.5x 20-period average) to filter signals.
# Designed to work in both bull and bear markets by identifying overextended conditions.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for multi-timeframe RSI
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate RSI for 1d timeframe
    def calculate_rsi(prices, period=14):
        if len(prices) < period:
            return np.full(len(prices), np.nan)
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = np.nan
        return rsi
    
    rsi_1d = calculate_rsi(close_1d, 14)
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Align RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_d = rsi_1d_aligned[i]
        rsi_w = rsi_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: 1d RSI oversold AND 1w RSI not overbought + volume confirmation
            if (rsi_d < 30 and rsi_w < 40 and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: 1d RSI overbought AND 1w RSI not oversold + volume confirmation
            elif (rsi_d > 70 and rsi_w > 60 and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 1d RSI returns to neutral territory
            if rsi_d > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 1d RSI returns to neutral territory
            if rsi_d < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_RSI_MeanReversion_Volume"
timeframe = "12h"
leverage = 1.0