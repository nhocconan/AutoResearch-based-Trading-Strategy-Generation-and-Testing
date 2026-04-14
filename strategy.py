#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian channel breakout with volume confirmation and ADX trend filter
# - Long when price breaks above previous 24h high with volume > 1.5x 48-period average and ADX > 25 (trending)
# - Short when price breaks below previous 24h low with volume > 1.5x 48-period average and ADX > 25 (trending)
# - Uses ADX > 25 to filter for trending markets and avoid whipsaws in ranging periods
# - Exits on opposite breakout (mean reversion tendency in ranging markets)
# - Position size 0.25 to balance risk and returns
# - Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag
# - Works in both bull and bear markets by capturing breakouts with trend confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX for trend filter (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 48-period average (2 days of 4h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous 1d high/low for breakout levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        
        # Create arrays for alignment (constant values for the 1d period)
        high_array = np.full(len(df_1d), prev_high)
        low_array = np.full(len(df_1d), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_1d, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_1d, low_array)[i]
        
        if position == 0:
            # Long: Break above previous 24h high with volume and trend filter
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                adx[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: Break below previous 24h low with volume and trend filter
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  adx[i] > 25):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous 24h low (mean reversion in ranging markets)
            if close[i] < low_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous 24h high (mean reversion in ranging markets)
            if close[i] > high_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_DonchianBreakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0