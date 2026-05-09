#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_KAMA_Trend_RSI_Reversal"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    """
    1d KAMA trend with weekly RSI reversal signals.
    - Long: KAMA trending up + RSI(14) < 30 on weekly timeframe
    - Short: KAMA trending down + RSI(14) > 70 on weekly timeframe
    - Exit: RSI crosses back above 50 (long) or below 50 (short)
    - Volume filter: current volume > 1.5 x 20-day average
    - Designed for 7-25 trades/year on 1d timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate KAMA(10) on daily data
    er_period = 10
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, n=1))
    change = np.insert(change, 0, np.nan)
    
    volatility = np.zeros(n)
    for i in range(1, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
    
    er = np.zeros(n)
    er[:] = np.nan
    for i in range(er_period, n):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[:] = np.nan
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) on weekly data
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    delta = np.insert(delta, 0, np.nan)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[:] = np.nan
    avg_loss[:] = np.nan
    
    # Wilder smoothing
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain[i] = np.nanmean(gain[1:15])
            avg_loss[i] = np.nanmean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(avg_gain)
    rs[:] = np.nan
    rsi_1w = np.zeros_like(avg_gain)
    rsi_1w[:] = np.nan
    
    for i in range(14, len(close_1w)):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi_1w[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi_1w[i] = 100 if avg_gain[i] > 0 else 0
    
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume filter (20-day average)
    vol_avg = np.zeros(n)
    vol_avg[:] = np.nan
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-day average
        vol_filter = volume[i] > vol_avg[i] * 1.5
        
        # KAMA trend direction
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        if position == 0:
            # Long: KAMA trending up + weekly RSI oversold + volume filter
            if (kama_up and rsi_1w_aligned[i] < 30 and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down + weekly RSI overbought + volume filter
            elif (kama_down and rsi_1w_aligned[i] > 70 and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly RSI crosses back above 50
            if rsi_1w_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly RSI crosses back below 50
            if rsi_1w_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals