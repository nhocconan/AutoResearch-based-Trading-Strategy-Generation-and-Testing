#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI divergence with 1d trend filter and volume confirmation.
# RSI detects overbought/oversold conditions; divergence signals momentum exhaustion.
# 1d EMA provides trend filter to trade with higher timeframe momentum.
# Volume confirms conviction in price moves.
# Target: 50-150 total trades over 4 years (12-38/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate RSI (14-period) on 4h data
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[i-rsi_period+1:i+1])
            avg_loss[i] = np.mean(loss[i-rsi_period+1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rsi = np.zeros(n)
    for i in range(rsi_period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA (21-period) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    alpha = 2 / (21 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema_1d[i] = close_1d[i]
        else:
            ema_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_1d[i-1]
    
    # Align daily EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi[i]
        ema_1d_val = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price above 1d EMA + volume confirmation
            if (rsi_val < 30 and 
                price > ema_1d_val and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) + price below 1d EMA + volume confirmation
            elif (rsi_val > 70 and 
                  price < ema_1d_val and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI > 50 (momentum fading) OR price below 1d EMA
            if (rsi_val > 50 or 
                price < ema_1d_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI < 50 (momentum fading) OR price above 1d EMA
            if (rsi_val < 50 or 
                price > ema_1d_val):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_Divergence_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0