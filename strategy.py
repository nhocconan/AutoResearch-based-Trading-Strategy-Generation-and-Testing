#!/usr/bin/env python3
"""
1d_KAMA_Trend_Regime_Filter_v1
Hypothesis: On daily timeframe, enter long when KAMA trend is up (close > KAMA) AND choppiness index < 42 (trending market) AND volume > 1.5x 20-day average volume. Enter short when KAMA trend is down (close < KAMA) AND choppiness index < 42 AND volume > 1.5x 20-day average volume. Uses discrete sizing (0.0, ±0.25) to limit fee drag. Weekly trend filter (EMA50) ensures alignment with higher timeframe momentum. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for indicators
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency Ratio
    change = np.abs(close_1d.diff(10).values)  # 10-period change
    volatility = np.abs(close_1d.diff(1).rolling(10).sum().values)  # 10-period volatility
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d.values)
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d.iloc[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    atr_14 = []
    for i in range(len(high_1d)):
        if i < 14:
            atr_14.append(np.nan)
        else:
            tr = np.maximum(high_1d[i] - low_1d[i],
                           np.maximum(np.abs(high_1d[i] - close_1d.iloc[i-1]),
                                     np.abs(low_1d[i] - close_1d.iloc[i-1])))
            atr_14.append(np.mean(atr_14[i-14:i]) * 13/14 + tr/14 if not np.isnan(atr_14[i-14]) else np.mean(atr_14[max(0,i-13):i]) + tr/14)
    atr_14 = np.array(atr_14)
    sum_atr_14 = np.nancumsum(atr_14)  # cumulative sum handling NaNs
    highest_high = np.maximum.accumulate(high_1d)
    lowest_low = np.minimum.accumulate(low_1d)
    chop = 100 * np.log10(sum_atr_14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) > 0, chop, 50)  # default to 50 when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    # Weekly trend filter (EMA50 on 1w data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (10+30), chop (14), volume MA (20), weekly EMA (50)
    start_idx = max(30, 14, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend conditions
        kama_uptrend = close[i] > kama_aligned[i]
        kama_downtrend = close[i] < kama_aligned[i]
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Regime filter: trending market (low chop)
        trending_market = chop_aligned[i] < 42
        
        if position == 0:
            # Long: KAMA uptrend + weekly uptrend + trending market + volume spike
            long_signal = kama_uptrend and weekly_uptrend and trending_market and volume_spike[i]
            
            # Short: KAMA downtrend + weekly downtrend + trending market + volume spike
            short_signal = kama_downtrend and weekly_downtrend and trending_market and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA downtrend OR chop > 50 (choppy) OR weekly trend change
            if kama_downtrend or chop_aligned[i] > 50 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA uptrend OR chop > 50 (choppy) OR weekly trend change
            if kama_uptrend or chop_aligned[i] > 50 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Regime_Filter_v1"
timeframe = "1d"
leverage = 1.0