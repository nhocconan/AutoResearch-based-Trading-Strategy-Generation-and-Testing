#!/usr/bin/env python3
"""
6h_ADX_GoldenCross_Volume_V1
Strategy: 6h ADX trend strength combined with EMA golden cross and volume confirmation.
Long: ADX > 25 (strong trend) + EMA12 crosses above EMA26 + volume > 1.5x average
Short: ADX > 25 + EMA12 crosses below EMA26 + volume > 1.5x average
Exit: ADX < 20 (weak trend) or opposite crossover
Designed for 6h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via ADX trend filter - only trades when trend is strong.
"""

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
    
    # Get daily data for weekly context
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA12 and EMA26 on 6h data
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Calculate ADX (14-period) on 6h data
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    high_low = high - low
    high_prev_close = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    low_prev_close = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    
    # Directional movement
    up_move = np.concatenate([[0], high[1:] - high[:-1]])
    down_move = np.concatenate([[0], low[:-1] - low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA to 6h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema12[i]) or np.isnan(ema26[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        above_weekly_ema = close[i] > ema50_1w_aligned[i]
        below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # EMA crossover
        ema12_above = ema12[i] > ema26[i]
        ema12_below = ema12[i] < ema26[i]
        ema12_prev_above = ema12[i-1] > ema26[i-1] if i > 0 else False
        ema12_prev_below = ema12[i-1] < ema26[i-1] if i > 0 else False
        
        # Golden cross and death cross
        golden_cross = ema12_above and ema12_prev_below
        death_cross = ema12_below and ema12_prev_above
        
        if position == 0:
            # Long: strong trend + golden cross + volume + above weekly EMA
            if strong_trend and golden_cross and vol_confirm and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: strong trend + death cross + volume + below weekly EMA
            elif strong_trend and death_cross and vol_confirm and below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weak trend or death cross or below weekly EMA
            if weak_trend or death_cross or below_weekly_ema:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weak trend or golden cross or above weekly EMA
            if weak_trend or golden_cross or above_weekly_ema:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_GoldenCross_Volume_V1"
timeframe = "6h"
leverage = 1.0