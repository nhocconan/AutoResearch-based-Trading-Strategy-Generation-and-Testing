#!/usr/bin/env python3
"""
6h_1w_1d_OrderBlockBreakout
Hypothesis: Institutional order blocks form at weekly supply/demand zones (strong rejections). 
Price breaking above weekly demand zone (bullish OB) or below supply zone (bearish OB) 
with volume confirmation and aligned daily trend captures impulsive moves. 
Works in bull markets (continuation breaks) and bear markets (breakdowns). 
Targets 15-25 trades/year by requiring multi-timeframe confluence.
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
    
    # Get weekly data for order blocks (supply/demand zones)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Identify weekly bullish order blocks (demand zones): strong close after down move
    # Bearish order blocks (supply zones): strong close after up move
    body_size = np.abs(close_1w - open_1w) if 'open' in df_1w else np.abs(close_1w - np.roll(close_1w, 1))
    if 'open' not in df_1w:
        open_1w = np.roll(close_1w, 1)
        open_1w[0] = close_1w[0]
    body_size = np.abs(close_1w - open_1w)
    candle_range = high_1w - low_1w
    strong_close = body_size > (candle_range * 0.6)  # Strong close >60% of range
    
    # Bullish OB: strong green candle after down day (close > open and prior close < prior open)
    bullish_ob = strong_close & (close_1w > open_1w) & (np.roll(close_1w, 1) < np.roll(open_1w, 1))
    # Bearish OB: strong red candle after up day (close < open and prior close > prior open)
    bearish_ob = strong_close & (close_1w < open_1w) & (np.roll(close_1w, 1) > np.roll(open_1w, 1))
    
    # OB zones: use the candle's high/low as the zone
    bullish_ob_low = np.where(bullish_ob, low_1w, np.nan)
    bullish_ob_high = np.where(bullish_ob, high_1w, np.nan)
    bearish_ob_low = np.where(bearish_ob, low_1w, np.nan)
    bearish_ob_high = np.where(bearish_ob, high_1w, np.nan)
    
    # Forward fill to create persistent zones until broken
    def ffill_np(arr):
        mask = np.isnan(arr)
        idx = np.where(~mask, np.arange(len(arr)), 0)
        np.maximum.accumulate(idx, out=idx)
        return arr[idx]
    
    bullish_ob_low_ff = ffill_np(bullish_ob_low)
    bullish_ob_high_ff = ffill_np(bullish_ob_high)
    bearish_ob_low_ff = ffill_np(bearish_ob_low)
    bearish_ob_high_ff = ffill_np(bearish_ob_high)
    
    # Get daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    open_1d = df_1d['open'].values if 'open' in df_1d else np.roll(close_1d, 1)
    
    # Volume expansion: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_expansion = volume_1d > (vol_ma_20 * 2.0)
    
    # Daily trend: price above/below 20 EMA
    close_series = pd.Series(close_1d)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend = close_1d > ema_20
    downtrend = close_1d < ema_20
    
    # Align all signals to 6h timeframe
    bullish_ob_low_aligned = align_htf_to_ltf(prices, df_1w, bullish_ob_low_ff)
    bullish_ob_high_aligned = align_htf_to_ltf(prices, df_1w, bullish_ob_high_ff)
    bearish_ob_low_aligned = align_htf_to_ltf(prices, df_1w, bearish_ob_low_ff)
    bearish_ob_high_aligned = align_htf_to_ltf(prices, df_1w, bearish_ob_high_ff)
    volume_expansion_aligned = align_htf_to_ltf(prices, df_1d, volume_expansion.astype(float))
    uptrend_aligned = align_htf_to_ltf(prices, df_1d, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(prices, df_1d, downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bullish_ob_low_aligned[i]) or 
            np.isnan(bullish_ob_high_aligned[i]) or 
            np.isnan(bearish_ob_low_aligned[i]) or 
            np.isnan(bearish_ob_high_aligned[i]) or 
            np.isnan(volume_expansion_aligned[i]) or 
            np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long break: price breaks above bullish OB zone (demand) with volume and uptrend
        long_break = close[i] > bullish_ob_high_aligned[i]
        long_entry = long_break and volume_expansion_aligned[i] > 0.5 and uptrend_aligned[i] > 0.5
        
        # Short break: price breaks below bearish OB zone (supply) with volume and downtrend
        short_break = close[i] < bearish_ob_low_aligned[i]
        short_entry = short_break and volume_expansion_aligned[i] > 0.5 and downtrend_aligned[i] > 0.5
        
        # Exit when price returns to opposite OB zone (mean reversion within the structure)
        exit_long = position == 1 and close[i] <= bullish_ob_low_aligned[i]
        exit_short = position == -1 and close[i] >= bearish_ob_high_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_OrderBlockBreakout"
timeframe = "6h"
leverage = 1.0