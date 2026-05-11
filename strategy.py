#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Price breaking above daily R3 or below daily S3 with 1-day trend confirmation and volume spike. Uses daily pivot levels as strong support/resistance. In uptrend, buy breakouts above R3; in downtrend, sell breakdowns below S3. Volume confirms institutional interest. Designed for low frequency (15-35 trades/year) with high win rate in trending markets. Works in both bull (breakouts) and bear (breakdowns) markets. Focus on BTC and ETH as primary targets.
"""

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # Using previous day's OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Shift to use previous day's data (avoid look-ahead)
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    # First day: use current values to avoid NaN
    d_high_prev[0] = d_high[0]
    d_low_prev[0] = d_low[0]
    d_close_prev[0] = d_close[0]
    
    # Pivot point calculation
    pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    # R3 and S3 levels
    r3 = pivot + 2 * (d_high_prev - d_low_prev)
    s3 = pivot - 2 * (d_high_prev - d_low_prev)
    
    # Align daily R3/S3 to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1-day trend filter (EMA 34)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 4h = ~3.3 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above daily R3 + above 1d EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S3 + below 1d EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to daily pivot or trend reversal
            if position == 1:
                # Exit long: price returns to daily pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals