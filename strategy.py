#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with Donchian(20) breakout
# In trending markets (CHOP < 38.2): follow Donchian breakout
# In ranging markets (CHOP > 61.8): fade Donchian breakout (mean reversion)
# Volume confirmation: volume > 1.5x 20-period average
# Stoploss: 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_chop_regime_donchian_vol_v4"
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
    
    # 1-day data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Choppiness Index (14-period) on daily data
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
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = hh_14d - ll_14d
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    # Choppiness Index formula
    chop = 100 * np.log10(tr_sum / range_14) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Neutral when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 20-period Donchian channels on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
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
            np.isnan(volume_ma[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(atr[i])):
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
            # Exit conditions based on regime
            elif chop_aligned[i] > 61.8 and close[i] < lowest_low[i]:  # Range: fade at support
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_aligned[i] < 38.2 and close[i] < highest_high[i]:  # Trend: exit on breakdown
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
            # Exit conditions based on regime
            elif chop_aligned[i] > 61.8 and close[i] > highest_high[i]:  # Range: fade at resistance
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            elif chop_aligned[i] < 38.2 and close[i] > lowest_low[i]:  # Trend: exit on breakout
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with regime filter
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            
            # Regime-based entry logic
            if chop_aligned[i] < 38.2:  # Trending market
                # Follow the trend: breakout entries
                if close[i] > highest_high[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < lowest_low[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            elif chop_aligned[i] > 61.8:  # Ranging market
                # Fade the move: mean reversion at extremes
                if close[i] < lowest_low[i] and volume_filter:
                    signals[i] = 0.25  # Buy at support
                    position = 1
                    entry_price = close[i]
                elif close[i] > highest_high[i] and volume_filter:
                    signals[i] = -0.25  # Sell at resistance
                    position = -1
                    entry_price = close[i]
    
    return signals