#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX breakout with 1d EMA filter and volume confirmation.
# ADX > 25 indicates strong trend on 12h.
# Price > 1d EMA50 for long, < 1d EMA50 for short ensures alignment with daily trend.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (trend up) and bear markets (trend down).
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
name = "12h_ADXBreakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX on 12h data
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    close_12h = pd.Series(df_12h['close'].values)
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = abs(high_12h - close_12h.shift(1))
    tr3 = abs(low_12h - close_12h.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_12h.diff()
    down_move = low_12h.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_12h)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_12h = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to lower timeframe (12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Get 1d data for EMA50 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_12h_aligned[i]
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long: Strong trend AND price > 1d EMA50 AND volume spike
            if strong_trend and price > ema_50 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Strong trend AND price < 1d EMA50 AND volume spike
            elif strong_trend and price < ema_50 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Trend weakens OR price crosses below EMA50
            if adx_12h_aligned[i] < 20 or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Trend weakens OR price crosses above EMA50
            if adx_12h_aligned[i] < 20 or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals