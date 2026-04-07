#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and Choppiness regime filter
# Long when price breaks above Donchian(20) high + 1-day volume > 1.5x 20-day average + Choppiness > 61.8 (range)
# Short when price breaks below Donchian(20) low + 1-day volume > 1.5x 20-day average + Choppiness > 61.8
# Exit when price returns to Donchian midpoint or Choppiness < 38.2 (trending)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 100-200 total trades over 4 years (25-50/year)

name = "12h_donchian20_1d_vol_chop_v1"
timeframe = "12h"
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
    
    # 1-day data for volume and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-day)
    vol_1d = df_1d['volume'].values
    vol_1d_s = pd.Series(vol_1d)
    vol_ma = vol_1d_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1-day Choppiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh_14 - ll_14)) / np.log10(14)
    
    # Align 1-day data to 12h
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 12-hour Donchian(20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 12-hour ATR(14) for stoploss
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
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
            # Exit: price returns to Donchian midpoint or trending regime (Choppiness < 38.2)
            elif close[i] >= donchian_mid[i] or chop_aligned[i] < 38.2:
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
            # Exit: price returns to Donchian midpoint or trending regime
            elif close[i] <= donchian_mid[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume confirmation + range regime
            # Volume confirmation: 1-day volume > 1.5x 20-day average
            vol_confirm = vol_ma_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_aligned[i]
            # Range regime: Choppiness > 61.8 (range-bound market)
            range_regime = chop_aligned[i] > 61.8
            
            # Long: break above Donchian high + volume confirmation + range regime
            if close[i] > highest_high[i] and vol_confirm and range_regime:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: break below Donchian low + volume confirmation + range regime
            elif close[i] < lowest_low[i] and vol_confirm and range_regime:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals