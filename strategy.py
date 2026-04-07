#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day volume confirmation and Choppiness regime filter
# Long when price breaks above Donchian(20) upper + volume > 1.5x 1-day avg volume + Choppiness > 61.8 (range) for mean reversion
# Short when price breaks below Donchian(20) lower + volume > 1.5x 1-day avg volume + Choppiness > 61.8
# Exit when price reverts to Donchian midpoint or Choppiness < 38.2 (trend)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day volume for confirmation and 4h Choppiness for regime filtering
# Target: 100-200 total trades over 4 years (25-50/year)

name = "4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_1d_s = pd.Series(vol_1d)
    vol_avg_1d = vol_1d_s.rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4-hour Choppiness Index for regime filter
    def calculate_chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        denom = (max_high - min_low)
        chop = np.where(denom != 0, 100 * np.log10(atr.sum() / denom) / np.log10(period), 50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i])):
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
            # Exit: price reverts to midpoint or trend regime (Choppiness < 38.2)
            elif close[i] >= donch_mid[i] or chop[i] < 38.2:
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
            # Exit: price reverts to midpoint or trend regime (Choppiness < 38.2)
            elif close[i] <= donch_mid[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and range regime
            volume_confirm = volume[i] > 1.5 * vol_avg_1d_aligned[i]
            range_regime = chop[i] > 61.8  # Chop > 61.8 = ranging market (good for mean reversion)
            
            # Long: price breaks above Donchian upper + volume confirmation + range regime
            if close[i] > donch_high[i] and volume_confirm and range_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower + volume confirmation + range regime
            elif close[i] < donch_low[i] and volume_confirm and range_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals