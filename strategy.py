#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_With_Volume_And_Chop_Filter"
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
    
    # Get daily data for KAMA and Chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close(t) - close(t-10)|
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # sum of absolute daily changes
    # Pad volatility array to match change length
    volatility = np.concatenate([np.full(9, np.nan), volatility[9:]])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # for EMA(2)
    slow_sc = 2 / (30 + 1) # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Calculate Choppiness Index on daily
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr1 = np.abs(high_arr[1:] - low_arr[1:])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr[1:] = tr
        # Wilder's smoothing
        atr_smoothed = np.zeros_like(atr)
        if len(atr) >= period:
            atr_smoothed[period-1] = np.mean(atr[1:period])
            for i in range(period, len(atr)):
                atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + atr[i]) / period
        # Calculate Chop
        max_high = np.zeros_like(close_arr)
        min_low = np.zeros_like(close_arr)
        for i in range(len(close_arr)):
            if i >= period-1:
                max_high[i] = np.max(high_arr[i-period+1:i+1])
                min_low[i] = np.min(low_arr[i-period+1:i+1])
            else:
                max_high[i] = np.nan
                min_low[i] = np.nan
        chop = np.full_like(close_arr, np.nan)
        for i in range(len(close_arr)):
            if i >= period-1 and not np.isnan(atr_smoothed[i]) and max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_smoothed[i] * period / (max_high[i] - min_low[i])) / np.log10(period)
        return chop
    
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Align KAMA and Chop to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection on 4h
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when market is trending (Chop < 38.2) or extreme chop (Chop > 61.8) for mean reversion
        # But for trend following, we use Chop < 38.2 (trending market)
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume spike filter
        
        if position == 0:
            # Long: Price above KAMA in trending market with volume
            if close[i] > kama_aligned[i] and chop_aligned[i] < 38.2 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA in trending market with volume
            elif close[i] < kama_aligned[i] and chop_aligned[i] < 38.2 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below KAMA or market becomes choppy
            if close[i] < kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above KAMA or market becomes choppy
            if close[i] > kama_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals