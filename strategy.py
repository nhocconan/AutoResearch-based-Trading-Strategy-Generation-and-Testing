#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v2
# Strategy: Daily Camarilla pivot reversal with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In ranging markets, price reverses at Camarilla pivot levels (H3/L3). 
# In trending markets, weekly EMA20 filters direction. Volume confirms institutional participation.
# Designed for low frequency (<25/year) to minimize fee drag in bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip first bar (need previous day for pivot)
        if i == 0:
            continue
            
        # Daily Camarilla pivot levels (based on previous day)
        phigh = high[i-1]
        plow = low[i-1]
        pclose = close[i-1]
        diff = phigh - plow
        
        # Camarilla levels
        h3 = pclose + (diff * 1.1 / 4)
        l3 = pclose - (diff * 1.1 / 4)
        h4 = pclose + (diff * 1.1 / 2)
        l4 = pclose - (diff * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if i >= 20:
            vol_avg_20 = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > (1.5 * vol_avg_20)
        else:
            vol_confirm = False
        
        # Weekly trend filter
        uptrend = ema_20_1w_aligned[i] > ema_20_1w_aligned[i-1] if not np.isnan(ema_20_1w_aligned[i]) else False
        downtrend = ema_20_1w_aligned[i] < ema_20_1w_aligned[i-1] if not np.isnan(ema_20_1w_aligned[i]) else False
        
        # Entry logic: reversal at H3/L3 with volume and weekly trend alignment
        if (close[i] <= h3 and close[i] > l3 and  # Between H3 and L3
            vol_confirm and 
            ((close[i] <= l3 and uptrend) or (close[i] >= h3 and downtrend)) and  # Reversal with trend
            position == 0):
            if close[i] <= l3:  # Near L3, go long in uptrend
                position = 1
                signals[i] = 0.25
            else:  # Near H3, go short in downtrend
                position = -1
                signals[i] = -0.25
        # Exit: price moves to H4/L4 or trend changes
        elif position == 1 and (close[i] >= h4 or not uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= l4 or not downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals