#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume spike and choppiness regime filter
# Donchian(20) breakout provides clear entry/exit levels
# Volume spike (>2.0x 20-period avg) confirms institutional participation
# Choppiness Index regime filter: CHOP > 61.8 = range (mean reversion at Donchian bounds)
#                            CHOP < 38.2 = trending (breakout continuation)
# Works in bull/bear markets: breakouts capture momentum in trends, mean reversion works in ranges
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Donchian20_1dVolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian(20) on 4h
    donch_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 14)  # warmup for Donchian, volume MA, Chop
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(vol_spike_aligned[i]) or np.isnan(chop_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol_spike = vol_spike_aligned[i] > 0.5
        curr_chop = chop_aligned[i]
        curr_donch_high = donch_high_20[i]
        curr_donch_low = donch_low_20[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume spike
            if curr_vol_spike:
                # Regime-based logic
                if curr_chop > 61.8:  # Range regime: mean reversion at Donchian bounds
                    # Long near lower bound, short near upper bound
                    if curr_low <= curr_donch_low * 1.001:  # touched lower bound
                        signals[i] = 0.25
                        position = 1
                    elif curr_high >= curr_donch_high * 0.999:  # touched upper bound
                        signals[i] = -0.25
                        position = -1
                elif curr_chop < 38.2:  # Trending regime: breakout continuation
                    # Breakout long/short
                    if curr_high > curr_donch_high:
                        signals[i] = 0.25
                        position = 1
                    elif curr_low < curr_donch_low:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price reaches opposite Donchian bound OR chop becomes extreme range
            if curr_low <= curr_donch_low * 1.001 or curr_chop > 80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price reaches opposite Donchian bound OR chop becomes extreme range
            if curr_high >= curr_donch_high * 0.999 or curr_chop > 80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals