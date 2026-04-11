#!/usr/bin/env python3
# 1h_4h_1d_volume_breakout_v1
# Strategy: 1-hour volume breakout with 4h/1d trend filter and volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Breakouts above rolling highs on volume spikes capture momentum in trending markets.
# Uses 4h EMA50 and 1d EMA200 for trend alignment to avoid counter-trend trades.
# Volume spike filter ensures institutional participation. Works in bull by riding trends,
# and in bear by catching mean-reversion bounces when volume spikes against trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volume_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for long-term trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 20-period rolling high/low for breakout detection
    roll_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    roll_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above/below 4h EMA50 and 1d EMA200
        uptrend_4h = price_close > ema_50_4h_aligned[i]
        uptrend_1d = price_close > ema_200_1d_aligned[i]
        downtrend_4h = price_close < ema_50_4h_aligned[i]
        downtrend_1d = price_close < ema_200_1d_aligned[i]
        
        # Breakout signals: price breaks rolling bands with volume
        long_breakout = (price_close > roll_high[i]) and vol_spike[i]
        short_breakout = (price_close < roll_low[i]) and vol_spike[i]
        
        # Exit when price returns to midpoint of range or opposite breakout
        mid_point = (roll_high[i] + roll_low[i]) / 2
        exit_long = position == 1 and (price_close < mid_point)
        exit_short = position == -1 and (price_close > mid_point)
        
        # Trading logic: require alignment of both timeframes
        if long_breakout and uptrend_4h and uptrend_1d and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_breakout and downtrend_4h and downtrend_1d and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals