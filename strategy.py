#!/usr/bin/env python3
# 6h_Fractal_Pivot_Reversal_v1
# Hypothesis: Williams fractals on 1d identify swing points. Price rejecting at fractal with volume spike indicates reversal. Fade at R3/S3 pivot levels in ranging markets, breakout continuation at R4/S4 in trending markets. Uses 12h EMA50 for trend filter. Designed for low frequency (12-37 trades/year) to avoid fee drag.

name = "6h_Fractal_Pivot_Reversal_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for fractals and pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Williams fractals on 1d (need 2-bar confirmation after center)
    bearish, bullish = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    bearish_fractal = align_htf_to_ltf(prices, df_1d, bearish, additional_delay_bars=2)
    bullish_fractal = align_htf_to_ltf(prices, df_1d, bullish, additional_delay_bars=2)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Daily pivot levels (standard)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3
    r3 = pivot + 2 * (daily_high - daily_low)
    s3 = pivot - 2 * (daily_high - daily_low)
    r4 = pivot + 3 * (daily_high - daily_low)
    s4 = pivot - 3 * (daily_high - daily_low)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation (50-period on 6h = ~12.5 days)
    volume_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need fractals (5), EMA50 (50), volume MA (50)
    start_idx = max(50, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bearish_fractal[i]) or 
            np.isnan(bullish_fractal[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (spike > 2.0x MA)
        volume_confirm = volume[i] > volume_ma[i] * 2.0
        
        if position == 0:
            # Long setup: bullish fractal rejection at S3/S4 + volume + not strong downtrend
            long_setup = (bullish_fractal[i] and 
                         (close[i] < s3_aligned[i] * 1.005 or close[i] < s4_aligned[i] * 1.005) and
                         volume_confirm and
                         not downtrend)  # Avoid strong downtrend
            
            # Short setup: bearish fractal rejection at R3/R4 + volume + not strong uptrend
            short_setup = (bearish_fractal[i] and 
                          (close[i] > r3_aligned[i] * 0.995 or close[i] > r4_aligned[i] * 0.995) and
                          volume_confirm and
                          not uptrend)  # Avoid strong uptrend
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: bearish fractal at resistance or trend breakdown
            if bearish_fractal[i] and close[i] > r3_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            elif not uptrend:  # Trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: bullish fractal at support or trend breakdown
            if bullish_fractal[i] and close[i] < s3_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            elif not downtrend:  # Trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals