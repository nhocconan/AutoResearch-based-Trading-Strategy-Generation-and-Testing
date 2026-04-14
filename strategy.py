#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Bollinger Bands squeeze breakout with 1-day ADX trend filter and volume confirmation.
# Bollinger Bands squeeze (low volatility) precedes breakouts in both bull and bear markets.
# ADX > 25 on daily timeframe confirms trending market, avoiding false breakouts in ranging conditions.
# Volume > 2x 20-period average confirms institutional participation.
# Position size: 0.25 (25%) to manage drawdown during volatile periods.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ADX(14) for trend strength filter
    adx_len = 14
    if len(df_1d) < adx_len:
        return np.zeros(n)
    
    # Calculate ADX components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=adx_len, adjust=False, min_periods=adx_len).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Bollinger Bands (20, 2) on 4h
    bb_len = 20
    bb_std = 2
    bb_middle = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_len, min_periods=bb_len).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze_condition = bb_width < 0.8 * bb_width_ma  # Bollinger Band squeeze
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, bb_len, 20, adx_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(squeeze_condition[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Enter long: Bollinger Band breakout above + squeeze + trending + volume
            if (close[i] > bb_upper[i] and 
                squeeze_condition[i] and 
                trending and 
                volume[i] > 2.0 * vol_ma[i]):
                position = 1
                signals[i] = position_size
            # Enter short: Bollinger Band breakout below + squeeze + trending + volume
            elif (close[i] < bb_lower[i] and 
                  squeeze_condition[i] and 
                  trending and 
                  volume[i] > 2.0 * vol_ma[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band
            if close[i] < bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band
            if close[i] > bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Bollinger_Squeeze_ADX_Volume_v1"
timeframe = "4h"
leverage = 1.0