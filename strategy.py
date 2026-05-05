#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Long when: Price breaks above Donchian upper (20) AND 1d ATR(14)/ATR(50) > 0.8 (low volatility regime) AND 4h volume > 1.5x 20-period average
# Short when: Price breaks below Donchian lower (20) AND 1d ATR(14)/ATR(50) > 0.8 AND 4h volume > 1.5x 20-period average
# Exit when price touches Donchian middle (20-period average) or opposite Donchian level
# Donchian provides clear breakout levels, ATR regime filter ensures we trade in low volatility environments where breakouts work better,
# Volume confirmation reduces false breakouts. Target: 80-150 total trades over 4 years (20-38/year) with discrete sizing 0.25

name = "4h_Donchian20_ATRRegime_Volume"
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
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    # ATR regime: low volatility when short ATR < long ATR * 0.8
    atr_regime = atr_14 > (atr_50 * 0.8)  # True when volatility is relatively low
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate 4h Donchian channels (20-period)
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    # Donchian middle = (upper + lower) / 2
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll_max
    donchian_lower = low_roll_min
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 4h volume confirmation (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr_regime_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_cond = bool(vol_spike[i])
        regime_cond = bool(atr_regime_aligned[i])
        
        if position == 0:
            # Long: Break above Donchian upper in low vol regime with volume spike
            if close[i] > donchian_upper[i] and regime_cond and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower in low vol regime with volume spike
            elif close[i] < donchian_lower[i] and regime_cond and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: touch Donchian middle or break below Donchian lower (reversal)
            if close[i] <= donchian_middle[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: touch Donchian middle or break above Donchian upper (reversal)
            if close[i] >= donchian_middle[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals