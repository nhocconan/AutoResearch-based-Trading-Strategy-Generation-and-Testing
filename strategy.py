#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA for trend (daily)
    if len(close_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = np.full_like(close_1d, np.nan)
    alpha = 2 / (34 + 1)
    ema34_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # Align EMA to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 14-period RSI for momentum (daily)
    if len(close_1d) < 14:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    rsi_14 = np.full_like(close_1d, np.nan)
    for i in range(13, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_14[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_14[i] = 100 if avg_gain[i] > 0 else 0
    
    # Align RSI to 12h timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 14-period ATR for volatility (daily)
    tr = np.zeros_like(high_1d)
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - high_1d[i-1]),
                   abs(low_1d[i] - low_1d[i-1]))
    
    atr_14 = np.full_like(high_1d, np.nan)
    if len(high_1d) >= 14:
        atr_14[13] = np.mean(tr[1:14])
        for i in range(14, len(high_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Position size: 25% of capital
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current period volume vs 14-period average
        vol_ma_14 = np.full_like(volume, np.nan)
        for j in range(13, len(volume)):
            vol_ma_14[j] = np.mean(volume[j-13:j+1])
        
        if np.isnan(vol_ma_14[i]) or vol_ma_14[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_14[i]
        
        if position == 0:
            # Long: Price above EMA34 + RSI > 50 + volume surge
            if (close[i] > ema34_1d_aligned[i] and
                rsi_14_aligned[i] > 50 and
                volume_ratio > 3.0):
                position = 1
                signals[i] = position_size
            # Short: Price below EMA34 + RSI < 50 + volume surge
            elif (close[i] < ema34_1d_aligned[i] and
                  rsi_14_aligned[i] < 50 and
                  volume_ratio > 3.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below EMA34
            if close[i] < ema34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above EMA34
            if close[i] > ema34_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_EMA_RSI_Volume_v1"
timeframe = "12h"
leverage = 1.0