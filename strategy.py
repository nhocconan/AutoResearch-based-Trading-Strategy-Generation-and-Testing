#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
# Donchian breakouts capture momentum; 1w Camarilla pivots provide institutional reference levels from weekly structure
# Volume confirmation ensures breakout authenticity
# Works in bull/bear: Weekly Camarilla pivots adapt to higher timeframe structure and reduce false breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels from prior week's OHLC
    # Camarilla levels: based on prior week's range
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Camarilla calculations
    camarilla_pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_range = prior_high - prior_low
    camarilla_h3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla data to 12h timeframe (wait for weekly close)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR price < Camarilla L3 (trend change vs Camarilla support)
            if close[i] < donchian_low[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR price > Camarilla H3 (trend change vs Camarilla resistance)
            if close[i] > donchian_high[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Camarilla filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND price > Camarilla H3 (bullish breakout above resistance)
                if close[i] > donchian_high[i] and close[i] > camarilla_h3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND price < Camarilla L3 (bearish breakdown below support)
                elif close[i] < donchian_low[i] and close[i] < camarilla_l3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals