#!/usr/bin/env python3
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
    
    # Load daily data for Bollinger Bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-day ATR
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    atr_1d = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 20:
        atr_1d[19] = np.mean(tr[1:20])
        for i in range(20, len(high_1d)):
            atr_1d[i] = (atr_1d[i-1] * 19 + tr[i]) / 20
    
    # Align daily ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 20-day SMA
    sma20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            sma20_1d[i] = np.mean(close_1d[i-19:i+1])
    
    sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma20_1d)
    
    # Calculate 20-day standard deviation
    std20_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 20:
        for i in range(19, len(close_1d)):
            std20_1d[i] = np.std(close_1d[i-19:i+1])
    
    std20_1d_aligned = align_htf_to_ltf(prices, df_1d, std20_1d)
    
    # Bollinger Bands (2 std dev)
    upper_bb_1d_aligned = sma20_1d_aligned + 2 * std20_1d_aligned
    lower_bb_1d_aligned = sma20_1d_aligned - 2 * std20_1d_aligned
    
    # Calculate 4h RSI with proper Wilder smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average (14-period)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
    
    # Wilder smoothing
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(close, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(sma20_1d_aligned[i]) or 
            np.isnan(std20_1d_aligned[i]) or
            np.isnan(upper_bb_1d_aligned[i]) or
            np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume surge and RSI oversold
            if (close[i] <= lower_bb_1d_aligned[i] and 
                volume_ratio > 2.0 and
                rsi[i] < 30):
                position = 1
                signals[i] = position_size
            # Short: price touches upper Bollinger Band with volume surge and RSI overbought
            elif (close[i] >= upper_bb_1d_aligned[i] and 
                  volume_ratio > 2.0 and
                  rsi[i] > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above SMA(20) or volume dries up or RSI overbought
            if (close[i] > sma20_1d_aligned[i] or
                volume_ratio < 0.5 or
                rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below SMA(20) or volume dries up or RSI oversold
            if (close[i] < sma20_1d_aligned[i] or
                volume_ratio < 0.5 or
                rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Touch_Volume_RSI"
timeframe = "4h"
leverage = 1.0