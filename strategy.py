#!/usr/bin/env python3
# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with 1-day trend filter and volume spike.
# Uses daily Camarilla levels for structure, 1-day EMA34 for trend direction,
# and volume > 1.5x 20-period average for confirmation. Designed for 20-40 trades/year
# to avoid fee drag while capturing strong trending moves in both bull and bear markets.
# Entry when price breaks R3 (long) or S3 (short) with trend alignment and volume confirmation.
# Exit when price returns to the Camarilla pivot point (central level) or reverses 50% of the breakout.

name = "4h_Camarilla_R3S3_Breakout_1DTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    # Camarilla levels
    r4 = c + range_val * 1.1/2
    r3 = c + range_val * 1.1/4
    r2 = c + range_val * 1.1/6
    r1 = c + range_val * 1.1/12
    pp = (h + l + c) / 3
    s1 = c - range_val * 1.1/12
    s2 = c - range_val * 1.1/6
    s3 = c - range_val * 1.1/4
    s4 = c - range_val * 1.1/2
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # We need to shift by 1 to avoid look-ahead (use previous day's data)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla for each day using previous day's OHLC
    pp = np.full_like(daily_close, np.nan)
    r3 = np.full_like(daily_close, np.nan)
    s3 = np.full_like(daily_close, np.nan)
    
    for i in range(1, len(daily_close)):
        pp[i], _, _, r3[i], _, _, _, s3[i], _ = calculate_camarilla(
            daily_high[i-1], daily_low[i-1], daily_close[i-1]
        )
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate daily EMA34 for trend filter (uses closed daily candle)
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after sufficient data for EMA34
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with uptrend and volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with downtrend and volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point or reverses 50% of breakout
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point or reverses 50% of breakout
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals