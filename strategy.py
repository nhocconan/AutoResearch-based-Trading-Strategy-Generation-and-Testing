#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
# Hypothesis: 6h Donchian breakouts capture medium-term momentum, filtered by weekly pivot trend
# (price above/below weekly pivot) and volume confirmation to avoid false breakouts.
# Works in bull (breakouts continue) and bear (failed breaks reverse) markets.
# Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
name = "6h_donchian20_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + weekly_range
    weekly_s2 = weekly_pivot - weekly_range
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed weeks only)
    pivot_6h = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_6h = align_htf_to_ltf(prices, df_1w, weekly_r1)
    s1_6h = align_htf_to_ltf(prices, df_1w, weekly_s1)
    r2_6h = align_htf_to_ltf(prices, df_1w, weekly_r2)
    s2_6h = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # 6h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly S1 or Donchian low breaks
            if close[i] < s1_6h[i] or close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above weekly R1 or Donchian high breaks
            if close[i] > r1_6h[i] or close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above Donchian high AND above weekly pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > pivot_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND below weekly pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < pivot_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals