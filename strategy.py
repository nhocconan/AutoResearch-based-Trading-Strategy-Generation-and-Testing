#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR(14) stoploss
# - Primary signal: 4h price breaks above Donchian(20) high for long, below Donchian(20) low for short
# - Volume confirmation: 1d volume > 1.5x 20-period median volume (avoid low-participation breakouts)
# - ATR stoploss: exit long when price < highest high since entry - 2.0*ATR(14), exit short when price > lowest low since entry + 2.0*ATR(14)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture strong moves, volume filter ensures participation, ATR stop manages risk in volatile markets

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    volume_1d = df_1d['volume'].values
    
    # 1d volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = volume_1d > (1.5 * median_volume_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_high = 0.0  # highest high since entry for longs
    entry_low = 0.0   # lowest low since entry for shorts
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            entry_high = max(entry_high, high[i])
            # Exit: price < entry_high - 2.0*ATR(14) OR Donchian breakout fails (price < lowest_low_20)
            if close[i] < entry_high - 2.0 * atr_14[i] or close[i] < lowest_low_20[i]:
                position = 0
                entry_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            entry_low = min(entry_low, low[i])
            # Exit: price > entry_low + 2.0*ATR(14) OR Donchian breakout fails (price > highest_high_20)
            if close[i] > entry_low + 2.0 * atr_14[i] or close[i] > highest_high_20[i]:
                position = 0
                entry_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long: price breaks above Donchian(20) high AND volume spike
            if close[i] > highest_high_20[i] and volume_spike_aligned[i]:
                position = 1
                entry_high = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low AND volume spike
            elif close[i] < lowest_low_20[i] and volume_spike_aligned[i]:
                position = -1
                entry_low = low[i]
                signals[i] = -0.25
    
    return signals