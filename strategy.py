#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 20-period Donchian breakouts with 1-week trend filter and volume confirmation
# Work in bull/bear: Trend filter ensures alignment with major trend; volume confirms breakout strength
# Target: 20-40 trades/year on daily timeframe to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly trend filter (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # === Daily Donchian channels ===
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(sma_50_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        trend = sma_50_1w_aligned[i]
        vol = vol_ratio[i]
        
        # Exit conditions
        if position == 1:  # Long
            if price < lower:  # Stop on opposite band break
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short
            if price > upper:  # Stop on opposite band break
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Long: Break above upper band, above weekly trend, strong volume
            if (price > upper) and (price > trend) and (vol > 1.8):
                signals[i] = 0.25
                position = 1
                continue
            # Short: Break below lower band, below weekly trend, strong volume
            elif (price < lower) and (price < trend) and (vol > 1.8):
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

name = "Daily_Donchian_20_WeeklyTrend_Filter_Volume"
timeframe = "1d"
leverage = 1.0