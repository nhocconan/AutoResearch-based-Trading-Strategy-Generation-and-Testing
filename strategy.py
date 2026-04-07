#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-day Donchian breakout
# Long when Donchian(20) upper break + CHOP(14) > 61.8 (range -> trend transition)
# Short when Donchian(20) lower break + CHOP(14) > 61.8
# Exit when CHOP < 38.2 (strong trend) or opposite Donchian break
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Donchian channels for breakout signals and 12h Choppiness for regime filter
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_chop_1d_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: 20-period high
    high_1d_s = pd.Series(high_1d)
    donch_high = high_1d_s.rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period low
    low_1d_s = pd.Series(low_1d)
    donch_low = low_1d_s.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # 12-period Choppiness Index (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    
    # 12-period ATR(14) for stoploss
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: strong trend (CHOP < 38.2) or opposite Donchian break
            elif chop[i] < 38.2 or close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: strong trend (CHOP < 38.2) or opposite Donchian break
            elif chop[i] < 38.2 or close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with choppy regime
            # Choppy regime: CHOP > 61.8 (range-bound, ready for breakout)
            choppy_regime = chop[i] > 61.8
            
            # Long: Donchian upper break + choppy regime
            if close[i] > donch_high_aligned[i] and choppy_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Donchian lower break + choppy regime
            elif close[i] < donch_low_aligned[i] and choppy_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals