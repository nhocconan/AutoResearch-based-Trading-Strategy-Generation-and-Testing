#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1w trend filter and chop regime filter.
- Trend filter: price > 1w EMA34 = bullish, price < 1w EMA34 = bearish.
- Regime filter: use 6h Chopiness Index (CHOP) to avoid ranging markets (CHOP > 61.8).
- In bullish 1w trend + low chop: buy breakouts above R3, sell breakdowns below S3.
- In bearish 1w trend + low chop: sell breakdowns below S3, buy breakouts above R3 (continuation logic).
- Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts.
- Exit on trend reversal, chop regime shift, or mean reversion to pivot.
- Position size: 0.25. Target: 50-150 total trades over 4 years = 12-37/year.
- Works in both bull and bear: 1w trend filter captures major moves, chop filter avoids whipsaws in ranging markets.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (using previous 1d bar's OHLC)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels (focus on R3/S3 for breakouts)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    pivot_val = pivot  # for mean reversion exit
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_val)
    
    # Calculate 6h Chopiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        """Calculate Chopiness Index"""
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                                   np.abs(low - np.roll(close, 1))))
        atr[0] = high[0] - low[0]  # first bar
        tr_sum = atr.rolling(window=window, min_periods=window).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min()
        chop = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop.values
    
    chop_values = calculate_chop(high, low, close, window=14)
    chop_threshold = 61.8  # above this = ranging market (avoid)
    low_chop = chop_values < chop_threshold
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34), CHOP(14), volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(chop_values[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend using EMA34
        htf_1w_bullish = close[i] > ema_34_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 1w trend with volume confirmation and low chop
            long_setup = (close[i] > r3_aligned[i]) and htf_1w_bullish and volume_spike[i] and low_chop[i]
            short_setup = (close[i] < s3_aligned[i]) and htf_1w_bearish and volume_spike[i] and low_chop[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal, chop regime shift (high chop), or mean reversion to pivot
            exit_signal = (not htf_1w_bullish) or (not low_chop[i]) or (close[i] < pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal, chop regime shift (high chop), or mean reversion to pivot
            exit_signal = htf_1w_bullish or (not low_chop[i]) or (close[i] > pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0