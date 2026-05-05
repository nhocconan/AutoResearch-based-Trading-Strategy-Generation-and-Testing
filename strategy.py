#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and choppiness regime filter
# Long when: Price breaks above 20-period 12h Donchian upper channel AND 1d volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# Short when: Price breaks below 20-period 12h Donchian lower channel AND 1d volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# Exit when price touches opposite Donchian channel (middle = 20-period average of high/low)
# Donchian channels provide clear price structure with proven edge in ranging markets
# Volume spike confirms institutional participation
# Chop regime filter (>61.8) ensures we trade in ranging markets where mean reversion works
# Target: 80-140 total trades over 4 years (20-35/year) with discrete sizing 0.25

name = "12h_Donchian20_VolumeSpike_Chop_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data ONCE before loop for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian(20) channels (based on previous 20 periods)
    # Upper = max(high) over last 20 periods, Lower = min(low) over last 20 periods
    # Middle = average of upper and lower
    high_roll_max = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Align 12h Donchian levels to 12h timeframe (no additional delay needed for breakout)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    
    # Calculate 1d volume spike (current volume > 1.8x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.8 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Calculate 1d True Range for chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, < 38.2 = trending
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Calculate sum of ATR14 over last 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    # Calculate max(high) - min(low) over last 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    # Avoid division by zero
    chop = 100 * (np.log10(sum_atr_14 / (14 * range_14)) / np.log10(14))
    # Handle invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50, chop)  # Default to neutral
    chop_regime = chop > 61.8  # Range regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike_aligned[i])
        chop_cond = bool(chop_regime_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian upper in range regime with volume spike
            if close[i] > donchian_upper_aligned[i] and vol_cond and chop_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower in range regime with volume spike
            elif close[i] < donchian_lower_aligned[i] and vol_cond and chop_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian middle or lower (reversal)
            if close[i] <= donchian_middle_aligned[i] or close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian middle or upper (reversal)
            if close[i] >= donchian_middle_aligned[i] or close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals