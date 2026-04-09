#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Donchian(20) breakouts capture strong momentum moves in both bull and bear markets
# 1d ADX > 25 ensures we only trade in trending regimes, avoiding choppy losses
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters false breakouts
# Fixed position size 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_1d_donchian_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channel (20-period)
    highest_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period) for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    highest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) - optional but helpful
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(highest_20_12h_aligned[i]) or np.isnan(lowest_20_12h_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: only trade when 1d ADX > 25 (trending market)
        trend_filter = adx_1d_aligned[i] > 25
        
        if not (volume_confirmed and trend_filter):
            signals[i] = 0.0
            continue
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to midpoint of Donchian channel
            midpoint = (highest_20_12h_aligned[i] + lowest_20_12h_aligned[i]) / 2
            if close[i] < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to midpoint of Donchian channel
            midpoint = (highest_20_12h_aligned[i] + lowest_20_12h_aligned[i]) / 2
            if close[i] > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Donchian breakout with volume and trend confirmation
            if close[i] > highest_20_12h_aligned[i]:  # Break above upper band -> long
                position = 1
                signals[i] = position_size
            elif close[i] < lowest_20_12h_aligned[i]:  # Break below lower band -> short
                position = -1
                signals[i] = -position_size
    
    return signals