#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Williams %R extreme levels with 12h ADX filter and volume confirmation.
# Long when daily Williams %R < -80 (oversold) with 12h ADX > 25 and volume > 1.3x 20-period average.
# Short when daily Williams %R > -20 (overbought) with 12h ADX > 25 and volume > 1.3x 20-period average.
# Exit when price crosses the 6h 20-period EMA (mean reversion) or Williams %R returns to -50 level.
# Uses discrete position size 0.25. Daily Williams %R provides contrarian signals from higher timeframe,
# 6h provides entry timing with ADX trend filter to avoid whipsaws.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Williams %R (14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align daily Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 12h data once before loop for ADX filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: ADX (14) for trend strength filter ===
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing)
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # 6h EMA (20) for exit signal
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Volume moving average (20-period) on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        price = close[i]
        ema_val = ema_20[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below 6h EMA(20) or Williams %R returns to -50
            if price < ema_val or wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above 6h EMA(20) or Williams %R returns to -50
            if price > ema_val or wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: only trade when ADX > 25 (strong trend)
            trend_filter = adx_val > 25
            
            # Volume filter: volume > 1.3x 20-period average
            vol_filter = vol > 1.3 * vol_ma
            
            # LONG: Daily Williams %R oversold (< -80) with trend and volume confirmation
            if (wr < -80) and trend_filter and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Daily Williams %R overbought (> -20) with trend and volume confirmation
            elif (wr > -20) and trend_filter and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_1dWilliamsR_12hADX_VolumeConfirmation_V1"
timeframe = "6h"
leverage = 1.0