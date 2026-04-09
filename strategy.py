#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume confirmation + ATR(14) stoploss
# - Primary signal: Donchian channel breakout on 4h timeframe - long when price > 20-period high, short when price < 20-period low
# - Volume filter: 1d volume > 20-period median volume (avoid low-participation breakouts)
# - ATR stoploss: exit when price moves against position by 2.0 * ATR(14) from entry
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture strong trends, volume filter ensures validity, ATR stop manages risk in volatile markets

name = "4h_1d_donchian_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d volume regime: volume > 20-period median
    volume_1d = df_1d['volume'].values
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_regime_1d = volume_1d > median_volume_20
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    # Pre-compute ATR(14) on 1d for stoploss
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute Donchian(20) on 4h (primary timeframe)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0  # Track entry price for ATR-based stoploss
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR ATR stoploss hit
            if close_4h[i] < lowest_low[i] or close_4h[i] < entry_price - 2.0 * atr_14_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR ATR stoploss hit
            if close_4h[i] > highest_high[i] or close_4h[i] > entry_price + 2.0 * atr_14_aligned[i]:
                position = 0
                entry_price = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high with volume confirmation
            if close_4h[i] > highest_high[i] and volume_regime_aligned[i]:
                position = 1
                entry_price = close_4h[i]
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low with volume confirmation
            elif close_4h[i] < lowest_low[i] and volume_regime_aligned[i]:
                position = -1
                entry_price = close_4h[i]
                signals[i] = -0.25
    
    return signals