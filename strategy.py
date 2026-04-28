#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike.
# Uses proven Camarilla structure from 12h pivots with 12h EMA50 for primary trend.
# Long when price breaks above R3 with volume and price > 12h EMA50 (uptrend).
# Short when price breaks below S3 with volume and price < 12h EMA50 (downtrend).
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 12h EMA50 trend filter.

name = "6h_Camarilla_R3S3_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot calculation and EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_prev = df_12h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_12h + low_12h + close_12h_prev) / 3.0
    # Range = H - L
    range_12h = high_12h - low_12h
    # Camarilla levels (R3/S3 are strong breakout levels)
    R3 = pivot + range_12h * 1.1 / 4.0
    S3 = pivot - range_12h * 1.1 / 4.0
    
    # Align to 6h timeframe (use previous 12h bar's levels)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume (stricter confirmation)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S3_aligned[i] or close[i] < ema_50_12h_aligned[i]
        short_exit = close[i] > R3_aligned[i] or close[i] > ema_50_12h_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals