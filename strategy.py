#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike + chop regime filter
# Long when: Price breaks above 12h Donchian upper channel (20) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Short when: Price breaks below 12h Donchian lower channel (20) AND 1d volume > 1.5x 20-period average AND chop > 61.8 (range regime)
# Exit when price returns to 12h Donchian midpoint (mean reversion in range)
# Donchian breakout captures volatility expansion after consolidation in ranging markets
# Volume spike confirms participation, chop filter ensures we only trade in ranging regimes
# Works in both bull and bear markets by trading mean reversion within ranges
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_DonchianBreakout_VolumeChop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume and chop regime filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range for chopiness index
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate chopiness index (14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log(max_high - min_low) * np.log(14)
    chop = 100 * np.log10(atr_sum / chop_denom) / np.log10(14)
    chop = np.where(chop_denom <= 0, 50, chop)  # Avoid division by zero/log of zero
    
    # Calculate 1d volume average (20)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h Donchian channels (20)
    donch_h_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_h_20 + donch_l_20) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donch_h_20[i]) or np.isnan(donch_l_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters: volume spike (>1.5x average) AND chop > 61.8 (range regime)
        vol_spike = volume[i] > (1.5 * vol_ma_20_aligned[i])
        range_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Break above upper Donchian channel in range regime with volume spike
            if close[i] > donch_h_20[i] and vol_spike and range_regime:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian channel in range regime with volume spike
            elif close[i] < donch_l_20[i] and vol_spike and range_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian midpoint (mean reversion)
            if close[i] < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian midpoint (mean reversion)
            if close[i] > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals