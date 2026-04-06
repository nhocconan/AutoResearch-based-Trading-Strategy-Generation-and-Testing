#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_13968_12h_1w_donchian_vol_v1"
timeframe = "12h"
leverage = 1.0

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period high with volume > 1.5x average and weekly close > weekly open.
# Short when price breaks below 20-period low with volume > 1.5x average and weekly close < weekly open.
# Exit when price returns to opposite Donchian band or weekly trend reverses.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Weekly trend filter reduces false breakouts in choppy markets.

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
    
    # Load weekly data for trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly trend: close > open = uptrend, close < open = downtrend
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_uptrend = weekly_close > weekly_open
    weekly_downtrend = weekly_close < weekly_open
    
    # Align weekly trend to 12h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # 12h data for Donchian channels, ATR, and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(volume_ma[i]) or \
           np.isnan(atr[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
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
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        # Donchian breakout signals
        breakout_high = close[i] > high_20[i-1]  # break above 20-period high
        breakdown_low = close[i] < low_20[i-1]   # break below 20-period low
        
        # Weekly trend filter
        weekly_up = weekly_uptrend_aligned[i-1] > 0.5  # previous week was uptrend
        weekly_down = weekly_downtrend_aligned[i-1] > 0.5  # previous week was downtrend
        
        # Entry signals
        long_signal = breakout_high and volume_ok and weekly_up
        short_signal = breakdown_low and volume_ok and weekly_down
        
        # Exit signals (return to opposite Donchian band or trend reversal)
        exit_long = close[i] < low_20[i] or weekly_downtrend_aligned[i] > 0.5
        exit_short = close[i] > high_20[i] or weekly_uptrend_aligned[i] > 0.5
        
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
            # Exit long on opposite band touch or trend reversal
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on opposite band touch or trend reversal
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals