#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13955_6d_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Hypothesis: 6h Donchian(20) breakout with 1-week pivot point bias and volume confirmation.
# Uses weekly Camarilla pivot levels (R3/S3, R4/S4) from prior week for bias:
# - Price > weekly R3: bullish bias (long only on breakouts)
# - Price < weekly S3: bearish bias (short only on breakouts)
# - Between S3/R3: no bias (await breakout in either direction with volume)
# Entry on 6h Donchian breakout in direction of bias with volume > 2x average.
# Exit on Donchian reversal or when price crosses opposite pivot level (S3/R3).
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Weekly pivot provides structural bias that works in both bull (buy strength) and bear (sell weakness).

def calculate_pivot_points(high, low, close):
    """Calculate Camarilla pivot points for given period"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 4.0)
    s3 = pivot - (range_val * 1.1 / 4.0)
    r4 = pivot + (range_val * 1.1 / 2.0)
    s4 = pivot - (range_val * 1.1 / 2.0)
    return pivot, r3, s3, r4, s4

def calculate_donchian(high, low, period):
    """Calculate Donchian upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using weekly close from prior week)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivots for each week
    wpivot, wr3, ws3, wr4, ws4 = calculate_pivot_points(weekly_high, weekly_low, weekly_close)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 week for look-ahead bias)
    wpivot_aligned = align_htf_to_ltf(prices, df_1w, wpivot)
    wr3_aligned = align_htf_to_ltf(prices, df_1w, wr3)
    ws3_aligned = align_htf_to_ltf(prices, df_1w, ws3)
    wr4_aligned = align_htf_to_ltf(prices, df_1w, wr4)
    ws4_aligned = align_htf_to_ltf(prices, df_1w, ws4)
    
    # 6h data for Donchian, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # ATR for stop loss
    atr = calculate_atr(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(50, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(wr3_aligned[i]) or np.isnan(ws3_aligned[i]) or \
           np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine bias from weekly Camarilla levels
        price = close[i]
        r3 = wr3_aligned[i]
        s3 = ws3_aligned[i]
        r4 = wr4_aligned[i]
        s4 = ws4_aligned[i]
        
        bullish_bias = price > r3  # Above R3: bullish bias
        bearish_bias = price < s3  # Below S3: bearish bias
        neutral_bias = (price >= s3) and (price <= r3)  # Between S3/R3: neutral
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 2.0)
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry signals
        if bullish_bias:
            # Only long on breakout up with volume
            long_signal = volume_ok and breakout_up
            short_signal = False
        elif bearish_bias:
            # Only short on breakout down with volume
            long_signal = False
            short_signal = volume_ok and breakout_down
        else:
            # Neutral: allow breakout in either direction with volume
            long_signal = volume_ok and breakout_up
            short_signal = volume_ok and breakout_down
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on Donchian breakdown or price crosses S3 (bearish bias)
            if close[i] < donchian_lower[i] or price < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on Donchian breakout or price crosses R3 (bullish bias)
            if close[i] > donchian_upper[i] or price > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals