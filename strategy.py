#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + choppiness regime filter
# - Primary signal: Price breaks above/below 4h Donchian channel (20-period high/low)
# - Volume confirmation: 1d volume > 1.5x 20-day median volume (avoid low-participation breakouts)
# - Regime filter: 1d Choppiness Index > 61.8 (range market) for mean reversion at channel edges
# - Exit: Price retracement to midpoint of Donchian channel
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Donchian breakouts capture trends, volume confirms validity, chop filter avoids false signals in strong trends

name = "4h_1d_donchian_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume regime: volume > 1.5x 20-period median volume
    median_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    volume_spike = volume_1d > (1.5 * median_volume_20)
    
    # 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    atr14 = np.zeros_like(close_1d)
    for i in range(14, len(close_1d)):
        atr14[i] = np.mean(np.column_stack([tr1[i-13:i+1], tr2[i-13:i+1], tr3[i-13:i+1]]), axis=1).sum()
    # Simplified: use rolling mean of true range
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * 14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(sum_atr14 / chop_denom)
    chop_regime = chop > 61.8  # range market
    
    # Align 1d indicators to 4h timeframe (completed 1d bar only)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 4h Donchian Channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(volume_spike_aligned[i]) or
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retracement to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with volume confirmation and chop regime filter
            # Long: price breaks above Donchian high AND volume spike AND chop regime (range)
            if close[i] > highest_high_20[i] and volume_spike_aligned[i] and chop_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND volume spike AND chop regime (range)
            elif close[i] < lowest_low_20[i] and volume_spike_aligned[i] and chop_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals