#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ATR regime filter
# Long when: Price breaks above Donchian upper (20) AND 1d volume > 1.5x 20-period average AND ATR(14) < ATR(50) (low vol regime)
# Short when: Price breaks below Donchian lower (20) AND 1d volume > 1.5x 20-period average AND ATR(14) < ATR(50)
# Exit when price returns to Donchian middle (mean of upper/lower)
# Donchian breakout captures volatility expansion after consolidation
# Volume spike confirms institutional participation
# ATR regime filter ensures we trade during low volatility periods (pre-breakout squeeze)
# Works in both bull and bear markets by trading breakouts in direction of the squeeze break
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_VolumeSpike_ATRRegime"
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
    
    # Get 1d data ONCE before loop for volume and ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50) and volume average
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d True Range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # Calculate 1d volume spike: current volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * vol_ma_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate Donchian Channels (20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR regime: low volatility (ATR14 < ATR50)
        low_vol_regime = atr_14_aligned[i] < atr_50_aligned[i]
        # Volume spike condition
        vol_spike = vol_spike_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Long: Break above Donchian high in low volatility regime with volume spike
            if close[i] > donchian_high[i] and low_vol_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in low volatility regime with volume spike
            elif close[i] < donchian_low[i] and low_vol_regime and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian middle (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian middle (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals