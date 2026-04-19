#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly trend filter + weekly pivot R1/S1 breakout with volume confirmation
# Weekly trend: price above/below weekly EMA34 determines bias
# Weekly pivot levels act as dynamic support/resistance
# Volume confirms breakout strength
# Target: 15-25 trades/year, works in bull/bear via trend filter

name = "1d_WeeklyTrend_Pivot_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high, low, close for EMA and pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly pivot point calculation (using weekly OHLC)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly R1 and S1 using Camarilla formula
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    # Align weekly data to daily timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema34 = ema34_1w_aligned[i]
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: weekly uptrend (price > EMA34) + break above weekly R1 + volume
            if price > ema34 and price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend (price < EMA34) + break below weekly S1 + volume
            elif price < ema34 and price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly downtrend or price breaks below weekly pivot
            if price < ema34 or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly uptrend or price breaks above weekly pivot
            if price > ema34 or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals