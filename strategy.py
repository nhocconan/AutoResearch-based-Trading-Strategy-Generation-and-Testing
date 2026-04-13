#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d chop regime filter.
    # Long when price breaks above Donchian(20) high and volume > 1.5x average and chop > 61.8 (range).
    # Short when price breaks below Donchian(20) low and volume > 1.5x average and chop > 61.8 (range).
    # Exit when price crosses Donchian(20) midpoint.
    # Uses volatility expansion in ranging markets to capture reversals with tight stops.
    # Target: 75-150 total trades over 4 years (19-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Get 1d data for chop regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Chopiness Index (14-period) on 1d
    atr_1d = np.zeros(len(close_1d))
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    # First TR is just high-low
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Calculate ATR(14) using Wilder's smoothing
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index = 100 * log10(sum(ATR)/log10(highest_high-lowest_low)) / log10(N)
    chop_denom = np.log10(highest_high_14 - lowest_low_14) * 14
    chop_num = np.log10(np.nansum(pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values))
    chop = 100 * chop_num / chop_denom
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Align HTF indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion breakouts)
        ranging_market = chop_aligned[i] > 61.8
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: price crosses Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions: breakout in ranging market with volume confirmation
        if long_breakout and ranging_market and volume_confirm and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and ranging_market and volume_confirm and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0