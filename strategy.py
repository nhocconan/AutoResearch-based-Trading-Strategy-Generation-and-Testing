#!/usr/bin/env python3
"""
4h_1d_1w_Trend_Reversal_Confirmation
Hypothesis: Combines daily trend alignment with weekly extreme reversals on 4h timeframe.
When daily trend is strong (ADX > 25), we look for reversals at weekly support/resistance.
When daily trend is weak (ADX < 25), we trade mean reversion at weekly extremes with volume confirmation.
Designed to capture both trending moves and range-bound reversals with controlled trade frequency.
"""

name = "4h_1d_1w_Trend_Reversal_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for ADX and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for weekly high/low
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d ADX for regime detection (14 period) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # --- 1d Close for trend direction ---
    trend_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # --- Weekly High/Low (using previous week) ---
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    
    # Align weekly levels to 4h
    weekly_high_4h = align_htf_to_ltf(prices, df_1w, prev_week_high)
    weekly_low_4h = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 40  # for ADX and trend
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(weekly_high_4h[i]) or np.isnan(weekly_low_4h[i]) or 
            np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Check stoploss (2.0x ATR from entry)
                atr_est = np.abs(high_4h[i] - low_4h[i])  # rough 4h ATR estimate
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine regime: ADX < 25 = range, ADX > 25 = trend
        is_range = adx_4h_aligned[i] < 25
        is_trend = adx_4h_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x 4h average
        vol_confirm = volume_4h[i] > 1.3 * vol_avg_4h[i]
        
        if position == 0:
            # Look for entries based on regime
            if is_range and vol_confirm:
                # Mean reversion: fade at weekly extremes
                if i > 0:
                    # Rejection at weekly high (failed breakout above)
                    if close_4h[i-1] > weekly_high_4h[i-1] and close_4h[i] < weekly_high_4h[i]:
                        signals[i] = -0.25  # short rejection
                        position = -1
                        entry_price = close_4h[i]
                    # Rejection at weekly low (failed breakdown below)
                    elif close_4h[i-1] < weekly_low_4h[i-1] and close_4h[i] > weekly_low_4h[i]:
                        signals[i] = 0.25   # long rejection
                        position = 1
                        entry_price = close_4h[i]
            elif is_trend and vol_confirm:
                # Trend following: look for reversals with trend confirmation
                if close_4h[i] > weekly_high_4h[i] and trend_1d_aligned[i] > close_4h[i]:
                    # Potential false breakout - look for reversal
                    if i > 0 and close_4h[i] < close_4h[i-1]:
                        signals[i] = -0.25  # short reversal
                        position = -1
                        entry_price = close_4h[i]
                elif close_4h[i] < weekly_low_4h[i] and trend_1d_aligned[i] < close_4h[i]:
                    # Potential false breakdown - look for reversal
                    if i > 0 and close_4h[i] > close_4h[i-1]:
                        signals[i] = 0.25   # long reversal
                        position = 1
                        entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if is_range:
                    # In range, take profit at weekly high or stop at weekly low
                    if close_4h[i] >= weekly_high_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below weekly low
                    elif close_4h[i] < weekly_low_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # is_trend
                    # In trend, trail with trend or stop at opposite weekly extreme
                    if close_4h[i] < trend_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below weekly low
                    elif close_4h[i] < weekly_low_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if is_range:
                    # In range, take profit at weekly low or stop at weekly high
                    if close_4h[i] <= weekly_low_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above weekly high
                    elif close_4h[i] > weekly_high_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # is_trend
                    # In trend, trail with trend or stop at opposite weekly extreme
                    if close_4h[i] > trend_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above weekly high
                    elif close_4h[i] > weekly_high_4h[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals