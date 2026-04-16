#!/usr/bin/env python3
"""
14d_Triple_SMA_Crossover_Volume
Hypothesis: On daily timeframe, a triple SMA (9/21/50) crossover with volume confirmation
captures medium-term trends while avoiding whipsaws. Long when SMA9 > SMA21 > SMA50 with
volume > 1.5x 20-day average; short when SMA9 < SMA21 < SMA50 with volume confirmation.
Uses 1w trend filter (close > 1w SMA200) to avoid counter-trend trades in strong trends.
Designed for low turnover (~15-25 trades/year) to minimize fee drag in 2025 bear market.
Works in both bull/bear: trend filter ensures alignment with higher timeframe momentum.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily SMA calculations ===
    close_series = pd.Series(close)
    sma9 = close_series.rolling(window=9, min_periods=9).mean().values
    sma21 = close_series.rolling(window=21, min_periods=21).mean().values
    sma50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    # === Volume confirmation (20-day average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma20
    
    # === 1-week trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # Align volume ratio to daily (already aligned as same timeframe)
    # But we'll keep it as is since it's calculated on daily data
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 200
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(sma9[i]) or np.isnan(sma21[i]) or np.isnan(sma50[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(sma200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Trend filter: only take longs in uptrend (price > 1w SMA200)
        uptrend = close[i] > sma200_1w_aligned[i]
        
        # Entry conditions
        bullish_alignment = sma9[i] > sma21[i] > sma50[i]
        bearish_alignment = sma9[i] < sma21[i] < sma50[i]
        vol_confirm = vol_ratio[i] > 1.5
        
        # Exit conditions
        if position == 1:  # Long
            # Exit: bearish alignment OR loss of uptrend
            if bearish_alignment or not uptrend:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            # Exit: bullish alignment OR gain of uptrend (in downtrend context)
            if bullish_alignment or uptrend:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry logic (only when flat)
        if position == 0:
            if bullish_alignment and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
                continue
            elif bearish_alignment and vol_confirm and not uptrend:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "14d_Triple_SMA_Crossover_Volume"
timeframe = "1d"
leverage = 1.0