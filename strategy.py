#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Donchian channel breakout with volume confirmation and ADX trend filter
# - Long when price breaks above previous weekly high with volume > 1.5x 20-day average and ADX > 25 (trending)
# - Short when price breaks below previous weekly low with volume > 1.5x 20-day average and ADX > 25 (trending)
# - Uses ADX > 25 to filter for trending markets and avoid whipsaws in ranging periods
# - Exits on opposite breakout (mean reversion tendency in ranging markets)
# - Position size 0.25 to balance risk and returns
# - Target: 40-100 trades over 4 years (10-25/year) to minimize fee drag
# - Works in both bull and bear markets by capturing breakouts with trend confirmation

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
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
    
    # Volume filter: 20-period average (approximately 20 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Get previous weekly high/low for breakout levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        
        # Create arrays for alignment (constant values for the 1w period)
        high_array = np.full(len(df_1w), prev_high)
        low_array = np.full(len(df_1w), prev_low)
        
        # Align to 1d timeframe
        high_1d = align_htf_to_ltf(prices, df_1w, high_array)[i]
        low_1d = align_htf_to_ltf(prices, df_1w, low_array)[i]
        
        if position == 0:
            # Long: Break above previous weekly high with volume and trend filter
            if (close[i] > high_1d and 
                volume[i] > vol_ma[i] * 1.5 and
                adx[i] > 25):
                position = 1
                signals[i] = position_size
            # Short: Break below previous weekly low with volume and trend filter
            elif (close[i] < low_1d and 
                  volume[i] > vol_ma[i] * 1.5 and
                  adx[i] > 25):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous weekly low (mean reversion in ranging markets)
            if close[i] < low_1d:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous weekly high (mean reversion in ranging markets)
            if close[i] > high_1d:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_DonchianBreakout_Volume_ADXFilter"
timeframe = "1d"
leverage = 1.0