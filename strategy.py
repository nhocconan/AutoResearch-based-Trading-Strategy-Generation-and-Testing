#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume_1wFilter
# Hypothesis: Camarilla R1/S1 breakout with 1d EMA trend filter, volume confirmation, and 1w trend filter works in both bull and bear markets by capturing institutional reversal points with trend alignment and avoiding choppy conditions. Weekly filter prevents trading against major trend, improving win rate and reducing whipsaws in volatile markets.
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_1wFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1w = close_1w > ema34_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # We need to shift the 1d data by 1 to avoid look-ahead
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(df_1d['high'].values, 1)
    low_1d_shifted = np.roll(df_1d['low'].values, 1)
    close_1d_shifted[0] = np.nan  # First value invalid after roll
    
    # Calculate Camarilla levels for each 1d bar, then align to 4h
    camarilla_r1 = (close_1d_shifted + (high_1d_shifted - low_1d_shifted) * 1.1 / 12)
    camarilla_s1 = (close_1d_shifted - (high_1d_shifted - low_1d_shifted) * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_up_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > R1 + 1d uptrend + 1w uptrend + volume confirmation
            if close[i] > camarilla_r1_aligned[i] and trend_up_1d_aligned[i] and trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + 1d downtrend + 1w downtrend + volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and not trend_up_1d_aligned[i] and not trend_up_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < S1 OR 1d trend turns down OR 1w trend turns down
            if close[i] < camarilla_s1_aligned[i] or not trend_up_1d_aligned[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > R1 OR 1d trend turns up OR 1w trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_up_1d_aligned[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals