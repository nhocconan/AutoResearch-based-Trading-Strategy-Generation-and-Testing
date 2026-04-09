#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume spike and chop regime filter
# - Uses 4h Donchian channel breakout (20-period) for entry signals
# - Volume confirmation: 1d ATR-normalized volume > 2.0 (volume spike relative to volatility)
# - Regime filter: 1d Choppiness Index > 61.8 for mean-reversion (fade extreme touches in range)
# - ATR(14) trailing stop at 2.0x ATR from extreme for risk control
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: 75-200 total trades over 4 years (19-50/year) per 4h strategy guidelines
# - Novelty: Combines Donchian breakout with 1d volume/chop filters to avoid false breakouts
# - Works in both bull/bear: Donchian adapts to volatility, chop filter prevents whipsaws in ranges

name = "4h_1d_donchian_volume_chop_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volume normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR-normalized volume (volume / ATR)
    vol_norm = volume_1d / np.where(atr_1d > 0, atr_1d, 1)
    avg_vol_norm_20 = pd.Series(vol_norm).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_norm > (2.0 * avg_vol_norm_20)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    atr_14_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14_sum / range_14) / np.log10(14), 50)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_regime = chop > 61.8  # True when ranging/markets suitable for mean reversion
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(atr[i]) or
            atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_since_entry:
                highest_since_entry = high[i]
            
            # Exit conditions: price retraces 2.0x ATR from high OR Chop regime + Donchian lower band touch (mean reversion)
            if low[i] <= highest_since_entry - (2.0 * atr[i]) or \
               (chop_regime_aligned[i] and low[i] <= lowest_low_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_since_entry:
                lowest_since_entry = low[i]
            
            # Exit conditions: price retraces 2.0x ATR from low OR Chop regime + Donchian upper band touch (mean reversion)
            if high[i] >= lowest_since_entry + (2.0 * atr[i]) or \
               (chop_regime_aligned[i] and high[i] >= highest_high_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation
            # Long: price breaks above Donchian upper band AND volume spike
            if high[i] >= highest_high_20[i] and volume_spike_aligned[i]:
                position = 1
                highest_since_entry = high[i]
                lowest_since_entry = high[i]
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band AND volume spike
            elif low[i] <= lowest_low_20[i] and volume_spike_aligned[i]:
                position = -1
                highest_since_entry = low[i]
                lowest_since_entry = low[i]
                signals[i] = -0.25
    
    return signals