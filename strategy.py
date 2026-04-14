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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day ATR for volatility (more stable than 10-day)
    if len(high_1d) < 20:
        return np.zeros(n)
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr0 = np.array([np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])])
    tr = np.concatenate([tr0, np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr20 = np.full_like(close_1d, np.nan)
    for i in range(19, len(tr)):
        if i == 19:
            atr20[i] = np.mean(tr[0:20])
        else:
            atr20[i] = (atr20[i-1] * 19 + tr[i]) / 20
    
    atr20_aligned = align_htf_to_ltf(prices, df_1d, atr20)
    
    # Calculate 50-day SMA for trend
    if len(close_1d) < 50:
        return np.zeros(n)
    
    sma50_1d = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        sma50_1d[i] = np.mean(close_1d[i-49:i+1])
    
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate 20-day RSI for momentum (smoother than 14-day)
    if len(close_1d) < 20:
        return np.zeros(n)
    
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 20:
        avg_gain[19] = np.mean(gain[1:20])
        avg_loss[19] = np.mean(loss[1:20])
        for i in range(20, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 19 + gain[i]) / 20
            avg_loss[i] = (avg_loss[i-1] * 19 + loss[i]) / 20
    
    rs = np.full_like(close_1d, np.nan)
    rsi20 = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        if avg_loss[i] > 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi20[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi20[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi20_aligned = align_htf_to_ltf(prices, df_1d, rsi20)
    
    # Calculate 48-period volume moving average for confirmation
    vol_ma_48 = np.full_like(volume, np.nan)
    for i in range(47, len(volume)):
        vol_ma_48[i] = np.mean(volume[i-47:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr20_aligned[i]) or 
            np.isnan(sma50_1d_aligned[i]) or 
            np.isnan(rsi20_aligned[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 48-period average
        if vol_ma_48[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_48[i]
        
        if position == 0:
            # Long: Price above 50-day SMA + RSI > 50 + volume surge
            if (close[i] > sma50_1d_aligned[i] and
                rsi20_aligned[i] > 50 and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price below 50-day SMA + RSI < 50 + volume surge
            elif (close[i] < sma50_1d_aligned[i] and
                  rsi20_aligned[i] < 50 and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below 50-day SMA OR RSI < 40
            if (close[i] < sma50_1d_aligned[i] or 
                rsi20_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above 50-day SMA OR RSI > 60
            if (close[i] > sma50_1d_aligned[i] or 
                rsi20_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_SMA50_RSI20_Volume_Filter"
timeframe = "4h"
leverage = 1.0