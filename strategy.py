# 6h_triple_screen_rsi_volume
# Strategy: Triple screen approach - 1d trend filter (RSI), 6h momentum (RSI), volume confirmation
# Works in both bull/bear: Trend filter adapts to market regime, momentum captures entries, volume filters false signals
# Uses 6-hour timeframe with 1-day trend filter for balanced trade frequency

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_triple_screen_rsi_volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1D trend filter - RSI on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.zeros_like(close_1d)
    rs[13:] = avg_gain[13:] / np.where(avg_loss[13:] == 0, 1e-10, avg_loss[13:])
    rsi_1d = np.zeros_like(close_1d)
    rsi_1d[13:] = 100 - (100 / (1 + rs[13:]))
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6H momentum - RSI(9) on 6-hour
    delta_6h = np.diff(close)
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    
    avg_gain_6h = np.zeros_like(close)
    avg_loss_6h = np.zeros_like(close)
    avg_gain_6h[8] = np.mean(gain_6h[1:9])
    avg_loss_6h[8] = np.mean(loss_6h[1:9])
    
    for i in range(9, len(close)):
        avg_gain_6h[i] = (avg_gain_6h[i-1] * 8 + gain_6h[i-1]) / 9
        avg_loss_6h[i] = (avg_loss_6h[i-1] * 8 + loss_6h[i-1]) / 9
    
    rs_6h = np.zeros_like(close)
    rs_6h[8:] = avg_gain_6h[8:] / np.where(avg_loss_6h[8:] == 0, 1e-10, avg_loss_6h[8:])
    rsi_6h = np.zeros_like(close)
    rsi_6h[8:] = 100 - (100 / (1 + rs_6h[8:]))
    
    # Volume filter - 1.5x 20-period average on 6h
    vol_ma = np.zeros_like(volume)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    vol_surge = np.zeros(n, dtype=bool)
    for i in range(19, n):
        if vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 14)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if required data not available
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_6h[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR volume dries up
            if rsi_6h[i] > 70 or not vol_surge[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI oversold OR volume dries up
            if rsi_6h[i] < 30 or not vol_surge[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Daily trend bullish (RSI > 50), 6h momentum bullish (RSI > 50), volume surge
            if (rsi_1d_aligned[i] > 50 and 
                rsi_6h[i] > 50 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Daily trend bearish (RSI < 50), 6h momentum bearish (RSI < 50), volume surge
            elif (rsi_1d_aligned[i] < 50 and 
                  rsi_6h[i] < 50 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals