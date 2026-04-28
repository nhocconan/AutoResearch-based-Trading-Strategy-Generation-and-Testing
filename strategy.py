#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike.
# Uses proven Camarilla structure from 4h pivots with 4h EMA50 for primary trend.
# Long when price breaks above R1 with volume and price > 4h EMA50 (uptrend).
# Short when price breaks below S1 with volume and price < 4h EMA50 (downtrend).
# Volume spike (>1.8x 20-bar average) confirms breakout strength.
# Session filter (08-20 UTC) reduces noise trades.
# Position size 0.20 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 4h EMA50 trend filter.

name = "1h_Camarilla_R1S1_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for pivot calculation and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_4h + low_4h + close_4h_prev) / 3.0
    # Range = H - L
    range_4h = high_4h - low_4h
    # Camarilla levels (R1/S1 are breakout levels)
    R1 = pivot + range_4h * 1.1 / 12.0
    S1 = pivot - range_4h * 1.1 / 12.0
    
    # Align to 1h timeframe (use previous 4h bar's levels)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Calculate 1h volume spike: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50 and pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R1_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S1_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < S1_aligned[i] or close[i] < ema_50_4h_aligned[i]
        short_exit = close[i] > R1_aligned[i] or close[i] > ema_50_4h_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals