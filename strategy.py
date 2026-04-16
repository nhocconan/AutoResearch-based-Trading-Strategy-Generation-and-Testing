# 1d_DonchianBreakout_1wTrendFilter_V1
# 1-day Donchian breakout with 1-week trend filter
# Long when price breaks above 20-period high AND 1-week EMA200 filter confirms uptrend
# Short when price breaks below 20-period low AND 1-week EMA200 filter confirms downtrend
# Trend filter: EMA200 > EMA200[1] for long, EMA200 < EMA200[1] for short
# Exit when price crosses back through the Donchian channel midpoint
# Target: 30-100 total trades over 4 years (7-25/year) with strong trend filtering

#!/usr/bin/env python3
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
    
    # === 1d Donchian channel (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # === 1w EMA200 for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or
            np.isnan(ema200_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = highest_20[i]
        lower = lowest_20[i]
        mid = donchian_mid[i]
        ema200 = ema200_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price crosses below Donchian midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above Donchian midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above upper band AND EMA200 trending up
            if price > upper and ema200 > ema200_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: price breaks below lower band AND EMA200 trending down
            elif price < lower and ema200 < ema200_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wTrendFilter_V1"
timeframe = "1d"
leverage = 1.0