#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_1dTrend_WeeklyVolatility"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly data for volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 13-day EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 1d trend: EMA34 slope
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Weekly ATR for volatility filter
    tr1w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), 
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr1w[0] = high_1w[0] - low_1w[0]
    atr14_1w = pd.Series(tr1w).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0 (market in balance with bullish bias),
            #       weekly ATR rising (increasing volatility), and 1d uptrend
            long_cond = (bull_power_aligned[i] > 0 and 
                        bear_power_aligned[i] < 0 and
                        atr14_1w_aligned[i] > atr14_1w_aligned[i-1] and
                        ema34_1d_aligned[i] > ema34_1d_aligned[i-1])
            
            # Short: Bear Power < 0, Bull Power < 0 (bearish bias),
            #        weekly ATR rising, and 1d downtrend
            short_cond = (bear_power_aligned[i] < 0 and 
                         bull_power_aligned[i] < 0 and
                         atr14_1w_aligned[i] > atr14_1w_aligned[i-1] and
                         ema34_1d_aligned[i] < ema34_1d_aligned[i-1])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power turns negative (momentum fading) OR 1d trend breaks
            if bear_power_aligned[i] < 0 or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power turns positive (momentum fading) OR 1d trend breaks
            if bull_power_aligned[i] > 0 or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals