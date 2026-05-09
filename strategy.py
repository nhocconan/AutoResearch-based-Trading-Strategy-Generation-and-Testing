#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1d ADX trend filter
# Long when price breaks above Donchian(20) high with ADX > 25 and volume > 1.5x average
# Short when price breaks below Donchian(20) low with ADX > 25 and volume > 1.5x average
# Exit when price crosses the Donchian(20) midline
# Uses volatility-based breakout with trend and volume filters to reduce false signals
# Designed for medium-frequency, high-conviction trades on 4h timeframe suitable for trending and ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) with size 0.25

name = "4h_DonchianBreakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels on 4h
    donchian_period = 20
    dc_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    dc_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    # Calculate ADX on 1d for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = pd.Series(df_1d['high']).subtract(df_1d['low']).abs()
    tr2 = pd.Series(df_1d['high']).subtract(df_1d['close'].shift(1)).abs()
    tr3 = pd.Series(df_1d['low']).subtract(df_1d['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).subtract(df_1d['high'].shift(1))
    down_move = pd.Series(df_1d['low'].shift(1)).subtract(df_1d['low'])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Directional Indicators
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_1d = dx.rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation on 1d
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 30)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high, ADX > 25, volume spike
            if (close[i] > dc_high[i] and adx_1d_aligned[i] > 25 and
                volume[i] > (1.5 * vol_ma_1d_aligned[i])):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, ADX > 25, volume spike
            elif (close[i] < dc_low[i] and adx_1d_aligned[i] > 25 and
                  volume[i] > (1.5 * vol_ma_1d_aligned[i])):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close[i] < dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close[i] > dc_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals