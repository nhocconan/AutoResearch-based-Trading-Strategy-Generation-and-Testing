#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness regime filter + Donchian(20) breakout with volume confirmation
# Long when price breaks above Donchian high + volume > 1.5x average + market is trending (CHOP < 38.2)
# Short when price breaks below Donchian low + volume > 1.5x average + market is trending (CHOP < 38.2)
# Exit when price crosses Donchian midpoint
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1-day Choppiness for regime filter and 1-day volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_chop_regime_donchian_vol_v2"
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
    
    # 1-day data for regime filter (Choppiness) and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Choppiness Index (14-period)
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
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr) / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop_raw = np.where(range_14 > 0, tr_sum / range_14, 1.0)
    chop_raw = np.maximum(chop_raw, 1e-10)  # prevent log(0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma)
    
    # 20-period Donchian channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian midpoint
            elif close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian midpoint
            elif close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and trending regime (CHOP < 38.2)
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Regime filter: market is trending (not choppy) - Choppiness < 38.2
            regime_filter = chop_aligned[i] < 38.2
            
            # Long: price breaks above Donchian high + volume filter + regime filter
            if close[i] > highest_high[i] and volume_filter and regime_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + regime filter
            elif close[i] < lowest_low[i] and volume_filter and regime_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals