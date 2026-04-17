#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with daily Williams Fractal breakout + volume confirmation + 1d ADX trend filter.
Long when price breaks above the most recent bullish fractal (high) with volume > 1.3x 20-period average and ADX > 20.
Short when price breaks below the most recent bearish fractal (low) with volume > 1.3x 20-period average and ADX > 20.
Williams Fractals identify key swing points where price has shown rejection; breakouts with volume and trend filter reduce false signals.
Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity and fee drag. Uses discrete sizing 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Fractals on daily data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    
    # For breakout signals, we need the most recent completed fractal levels
    # Initialize arrays to hold the most recent fractal levels
    recent_bearish = np.full_like(high_1d, np.nan)
    recent_bullish = np.full_like(low_1d, np.nan)
    
    # Track the most recent completed fractal
    last_bearish = np.nan
    last_bullish = np.nan
    
    for i in range(len(bearish_fractal)):
        if not np.isnan(bearish_fractal[i]):
            last_bearish = bearish_fractal[i]
        if not np.isnan(bullish_fractal[i]):
            last_bullish = bullish_fractal[i]
        recent_bearish[i] = last_bearish
        recent_bullish[i] = last_bullish
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 4h
    recent_bearish_aligned = align_htf_to_ltf(prices, df_1d, recent_bearish, additional_delay_bars=0)
    recent_bullish_aligned = align_htf_to_ltf(prices, df_1d, recent_bullish, additional_delay_bars=0)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(recent_bearish_aligned[i]) or np.isnan(recent_bullish_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 20 indicates sufficient trend strength
        trend_filter = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above most recent bullish fractal with volume and trend
            if (close[i] > recent_bullish_aligned[i] and 
                volume_confirmed and 
                trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below most recent bearish fractal with volume and trend
            elif (close[i] < recent_bearish_aligned[i] and 
                  volume_confirmed and 
                  trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below the most recent bullish fractal or trend weakens
            if (close[i] < recent_bullish_aligned[i] or 
                adx_aligned[i] < 15):  # exit when trend weakens significantly
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above the most recent bearish fractal or trend weakens
            if (close[i] > recent_bearish_aligned[i] or 
                adx_aligned[i] < 15):  # exit when trend weakens significantly
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dWilliamsFractal_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0