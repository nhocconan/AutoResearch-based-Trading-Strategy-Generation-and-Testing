#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R Extreme Reversal with 1d Volume Spike and Chop Regime Filter
# Long when: Williams %R(14) crosses above -80 (oversold) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Short when: Williams %R(14) crosses below -20 (overbought) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Exit when Williams %R returns to -50 (mean reversion center) or opposite extreme
# Williams %R identifies exhaustion points in ranging markets
# Volume spike confirms institutional participation at turning points
# Chop regime filter (>61.8) ensures we trade in ranging markets where mean reversion works
# Target: 100-180 total trades over 4 years (25-45/year) with discrete sizing 0.25

name = "4h_WilliamsR_Extreme_Reversal_1dVolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R, volume, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d True Range for chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, < 38.2 = trending
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero and invalid values
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    chop = np.where((range_14 == 0) | np.isnan(chop), 50, chop)  # Default to neutral
    chop_regime = chop > 61.8  # Range regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        wr = williams_r_aligned[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) in range regime with volume spike
            if wr > -80 and vol_cond and chop_cond:
                # Check for crossover: previous value was <= -80
                if i > 100 and williams_r_aligned[i-1] <= -80:
                    signals[i] = 0.25
                    position = 1
            # Short: Williams %R crosses below -20 (from above) in range regime with volume spike
            elif wr < -20 and vol_cond and chop_cond:
                # Check for crossover: previous value was >= -20
                if i > 100 and williams_r_aligned[i-1] >= -20:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 or crosses below -80 (reversal)
            if wr >= -50 or wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 or crosses above -20 (reversal)
            if wr <= -50 or wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals