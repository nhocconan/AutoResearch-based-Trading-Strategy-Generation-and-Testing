#!/usr/bin/env python3
"""
6h Williams %R + 1d ADX Trend + Volume Confirmation
Hypothesis: Williams %R identifies overbought/oversold conditions on 6h, while 1d ADX > 25 filters for trending markets.
In trending markets (ADX > 25), we fade extreme %R readings: long when %R < -80 (oversold) and price > 1d EMA50,
short when %R > -20 (overbought) and price < 1d EMA50. Volume confirmation (current volume > 1.5 * 20-bar average) ensures participation.
This combines mean reversion within a trend filter, working in both bull (buy dips) and bear (sell rallies) markets.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year on 6h.
"""

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
    
    # Get 1d data for ADX and EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX (14-period)
    # ADX requires +DI, -DI, and DX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # first period has no previous close
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)  # avoid division by zero
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 6h
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: current volume > 1.5 * 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        williams_r_val = williams_r[i]
        adx_val = adx_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: ADX > 25 (trending), Williams %R < -80 (oversold), price > 1d EMA50 (uptrend bias), volume spike
            long_entry = (adx_val > 25) and (williams_r_val < -80) and (curr_close > ema_trend) and vol_spike
            # Short: ADX > 25 (trending), Williams %R > -20 (overbought), price < 1d EMA50 (downtrend bias), volume spike
            short_entry = (adx_val > 25) and (williams_r_val > -20) and (curr_close < ema_trend) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R > -50 (exit oversold) OR ADX < 20 (trend weakening) OR price < 1d EMA50
            if (williams_r_val > -50) or (adx_val < 20) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R < -50 (exit overbought) OR ADX < 20 (trend weakening) OR price > 1d EMA50
            if (williams_r_val < -50) or (adx_val < 20) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_ADXTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0