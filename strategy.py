#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Donchian channel breakout with volume confirmation and rising ADX trend filter
# - Long when price breaks above previous weekly high with volume > 1.5x 48-period average and ADX > 20 (trending market)
# - Short when price breaks below previous weekly low with volume > 1.5x 48-period average and ADX > 20
# - Uses ADX > 20 to ensure trades occur only in trending conditions, avoiding whipsaws in ranging markets
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
    
    # Load 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate ADX for trend filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Volume filter: 48-period average (2 days of 4h bars)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            continue
        
        # Get previous 1w high/low for breakout levels
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        
        # Create arrays for alignment (constant values for the 1w period)
        high_array = np.full(len(df_1w), prev_high)
        low_array = np.full(len(df_1w), prev_low)
        
        # Align to 4h timeframe
        high_4h = align_htf_to_ltf(prices, df_1w, high_array)[i]
        low_4h = align_htf_to_ltf(prices, df_1w, low_array)[i]
        
        if position == 0:
            # Long: Break above previous weekly high with volume and trend filter
            if (close[i] > high_4h and 
                volume[i] > vol_ma[i] * 1.5 and
                adx[i] > 20):
                position = 1
                signals[i] = position_size
            # Short: Break below previous weekly low with volume and trend filter
            elif (close[i] < low_4h and 
                  volume[i] > vol_ma[i] * 1.5 and
                  adx[i] > 20):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: Break below previous weekly low (mean reversion in ranging markets)
            if close[i] < low_4h:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: Break above previous weekly high (mean reversion in ranging markets)
            if close[i] > high_4h:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1w_DonchianBreakout_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0