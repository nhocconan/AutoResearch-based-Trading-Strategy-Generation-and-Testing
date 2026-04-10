#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Trend up: Lips > Teeth > Jaw; Trend down: Lips < Teeth < Jaw
# - Enter long on Alligator uptrend + volume spike + chop < 61.8 (trending regime)
# - Enter short on Alligator downtrend + volume spike + chop < 61.8
# - Exit when Alligator trend reverses or chop > 61.8 (range regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in both bull/bear via trend-following with regime filter to avoid whipsaws

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator components (using median price)
    median_1d = (high_1d + low_1d) / 2
    jaw_period = 13
    jaw_shift = 8
    teeth_period = 8
    teeth_shift = 5
    lips_period = 5
    lips_shift = 3
    
    # Jaw: SMA(13,8) of median
    jaw_1d = pd.Series(median_1d).rolling(window=jaw_period, min_periods=jaw_period).mean().values
    jaw_1d = np.roll(jaw_1d, jaw_shift)
    jaw_1d[:jaw_shift] = np.nan
    
    # Teeth: SMA(8,5) of median
    teeth_1d = pd.Series(median_1d).rolling(window=teeth_period, min_periods=teeth_period).mean().values
    teeth_1d = np.roll(teeth_1d, teeth_shift)
    teeth_1d[:teeth_shift] = np.nan
    
    # Lips: SMA(5,3) of median
    lips_1d = pd.Series(median_1d).rolling(window=lips_period, min_periods=lips_period).mean().values
    lips_1d = np.roll(lips_1d, lips_shift)
    lips_1d[:lips_shift] = np.nan
    
    # Alligator alignment
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index (CHOP) - regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high,period) - min(low,period))) / log10(period)
    # CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    chop_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = np.zeros_like(tr)
    atr_14[chop_period-1] = np.mean(tr[:chop_period])
    for i in range(chop_period, len(tr)):
        atr_14[i] = (atr_14[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    sum_atr14 = pd.Series(atr_14).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    range_chop = max_high - min_low
    range_chop[range_chop == 0] = 1e-10
    
    chop_1d = 100 * np.log10(sum_atr14 / range_chop) / np.log10(chop_period)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(lips_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or np.isnan(jaw_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator trend reverses OR chop > 61.8 (range regime)
            if not (lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]) or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator trend reverses OR chop > 61.8 (range regime)
            if not (lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]) or chop_1d_aligned[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with volume spike and trending regime (chop < 61.8)
            if vol_spike_1d_aligned[i] and chop_1d_aligned[i] < 61.8:
                # Long signal: Alligator aligned up (Lips > Teeth > Jaw)
                if lips_1d_aligned[i] > teeth_1d_aligned[i] > jaw_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: Alligator aligned down (Lips < Teeth < Jaw)
                elif lips_1d_aligned[i] < teeth_1d_aligned[i] < jaw_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals