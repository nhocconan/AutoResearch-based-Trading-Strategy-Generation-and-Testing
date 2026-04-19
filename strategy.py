#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_Follow_With_1d_RSI_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI with proper Wilder's smoothing
    delta = np.diff(close_1d)
    delta = np.insert(delta, 0, np.nan)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_len = 14
    avg_up = np.full_like(close_1d, np.nan)
    avg_down = np.full_like(close_1d, np.nan)
    
    if len(up) >= rsi_len:
        avg_up[rsi_len-1] = np.nanmean(up[1:rsi_len+1])
        avg_down[rsi_len-1] = np.nanmean(down[1:rsi_len+1])
        
        for i in range(rsi_len, len(up)):
            avg_up[i] = (avg_up[i-1] * (rsi_len-1) + up[i]) / rsi_len
            avg_down[i] = (avg_down[i-1] * (rsi_len-1) + down[i]) / rsi_len
    
    rs = np.divide(avg_up, avg_down, out=np.full_like(avg_up, np.nan), where=avg_down!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate KAMA on 4h
    close_s = pd.Series(close)
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    change = np.insert(change, 0, np.nan)  # align with close
    volatility = np.abs(np.diff(close))
    volatility = np.insert(volatility, 0, np.nan)
    
    er = np.divide(change, np.nansum(volatility.reshape(-1, 10), axis=1), 
                   out=np.full_like(change, np.nan), where=~np.isnan(np.nansum(volatility.reshape(-1, 10), axis=1)))
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = np.nanmean(close[0:10])  # seed
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_1d_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # Trend filter: price above/below KAMA
        price_above_kama = price > kama[i]
        price_below_kama = price < kama[i]
        
        if position == 0:
            # Long: price > KAMA, RSI > 50 (bullish bias), with volume
            if price_above_kama and rsi > 50 and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI < 50 (bearish bias), with volume
            elif price_below_kama and rsi < 50 and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or RSI < 40
            if price < kama[i] or rsi < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or RSI > 60
            if price > kama[i] or rsi > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals