#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d Volume Spike and Donchian breakout.
# In choppy markets (CHOP > 61.8), fade Donchian breakouts (mean reversion).
# In trending markets (CHOP < 38.2), follow Donchian breakouts.
# Uses 1d volume spike (>2x 20-day avg) to confirm institutional participation.
# Position size: 0.25. Target: 20-35 trades/year.
# Works in bull/bear: chop regime adapts to market conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume spike filter ===
    df_1d = get_htf_data(prices, '1d')
    
    volume_1d = df_1d['volume'].values
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # === 4h Choppiness Index (14-period) ===
    def true_range(h, l, c_prev):
        return np.maximum(h - l, np.maximum(np.abs(h - c_prev), np.abs(l - c_prev)))
    
    tr14 = np.zeros(n)
    for i in range(1, n):
        tr14[i] = true_range(high[i], low[i], close[i-1])
    
    atr14 = pd.Series(tr14).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max(high, close_prev) and min(low, close_prev) over 14 periods
    max_hc = np.maximum(high, np.concatenate([[close[0]], close[:-1]]))
    min_lc = np.minimum(low, np.concatenate([[close[0]], close[:-1]]))
    
    max_range = pd.Series(max_hc - min_lc).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr14 * 14 / max_range) / np.log10(10)
    
    # === 4h Donchian Channels (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(chop[i]) or np.isnan(volume_ma20_1d_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned to 4h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > (2.0 * volume_ma20_1d_aligned[i])
        
        # Chop regime: >61.8 = range/chop, <38.2 = trending
        chop_value = chop[i]
        is_choppy = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        # Donchian breakout signals
        long_breakout = close[i] > highest_high[i-1]
        short_breakout = close[i] < lowest_low[i-1]
        
        if position == 0:
            # In choppy market: fade breakouts (mean reversion)
            if is_choppy and volume_filter:
                if short_breakout:  # faded short (sell breakdown)
                    signals[i] = 0.25
                    position = -1
                elif long_breakout:  # faded long (buy breakdown)
                    signals[i] = -0.25
                    position = 1
            # In trending market: follow breakouts
            elif is_trending and volume_filter:
                if long_breakout:
                    signals[i] = 0.25
                    position = 1
                elif short_breakout:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: opposite breakout or chop increases significantly
            if short_breakout or chop_value > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite breakout or chop increases significantly
            if long_breakout or chop_value > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Chop_Donchian_VolumeSpike"
timeframe = "4h"
leverage = 1.0