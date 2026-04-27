#!/usr/bin/env python3
"""
Hypothesis: 4-hour ATR breakout with volume confirmation and 1-day trend filter.
Trades only during high-momentum breakouts in the direction of the daily trend.
Designed to work in both bull and bear markets by using the daily trend as filter.
Target: 20-50 trades/year per symbol (80-200 total over 4 years) to minimize fee drag.
"""
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4-hour data for ATR and volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour ATR(14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all 4h indicators
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need ATR, volume MA, and 1d EMA
    start_idx = max(14, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        atr = atr_14_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Volume filter: volume > 1.5x 4h average (moderate to balance trades)
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Entry conditions: ATR breakout with volume and daily trend alignment
        if position == 0:
            # Long: break above close + 1*ATR + volume + daily uptrend
            if close[i] > close[i-1] + atr and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below close - 1*ATR + volume + daily downtrend
            elif close[i] < close[i-1] - atr and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below daily EMA or ATR-based stop
            if close[i] < trend_1d or close[i] < high[i-1] - atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above daily EMA or ATR-based stop
            if close[i] > trend_1d or close[i] > low[i-1] + atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ATRBreakout_Volume_1dTrendFilter"
timeframe = "4h"
leverage = 1.0