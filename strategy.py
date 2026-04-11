#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_breakout_v1
# Strategy: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels from the daily chart act as strong support/resistance.
# A breakout above/below these levels on the 12h chart, aligned with the daily trend
# and confirmed by volume, captures sustained moves in both bull and bear markets.
# Low trade frequency targets 12-37 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (Camarilla uses previous day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be invalid due to roll, but we'll handle with min_periods later
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Camarilla levels
    # Resistance levels
    r1 = pivot + (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    r3 = pivot + (range_ * 1.1 / 4)
    r4 = pivot + (range_ * 1.1 / 2)
    # Support levels
    s1 = pivot - (range_ * 1.1 / 12)
    s2 = pivot - (range_ * 1.1 / 6)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Price relative to Camarilla levels
        above_r3 = close[i] > r3_aligned[i]
        below_s3 = close[i] < s3_aligned[i]
        
        # Entry conditions
        # Long: Price crosses above R3 AND uptrend AND volume confirmation
        if above_r3 and uptrend and vol_confirm and position != 1:
            # Additional check: ensure we didn't just cross above R3 in previous bar
            if i == 50 or close[i-1] <= r3_aligned[i-1]:
                position = 1
                signals[i] = 0.25
        # Short: Price crosses below S3 AND downtrend AND volume confirmation
        elif below_s3 and downtrend and vol_confirm and position != -1:
            # Additional check: ensure we didn't just cross below S3 in previous bar
            if i == 50 or close[i-1] >= s3_aligned[i-1]:
                position = -1
                signals[i] = -0.25
        # Exit: Price crosses back through the pivot level (mean reversion signal)
        elif position == 1 and close[i] < pivot[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > pivot[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals