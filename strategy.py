#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume spike confirmation.
    # Long when price breaks above R4 with 1w EMA50 > EMA200 (uptrend) and volume > 2x average.
    # Short when price breaks below S4 with 1w EMA50 < EMA200 (downtrend) and volume > 2x average.
    # Exit when price retreats to R3/S3 or opposite pivot level.
    # Uses weekly trend to avoid counter-trend breakouts and volume spike for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    # We need to shift by 1 to avoid look-ahead: use previous day's data for today's levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on that day's OHLC)
    # Then align to 6h timeframe with proper delay
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    # R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low)
    # We use previous day's levels to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_r4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_s4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_r3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
    camarilla_s3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 and EMA200 on 1w
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate volume average (20-period) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA50 > EMA200 for uptrend, < for downtrend
        uptrend = ema50_1w_aligned[i] > ema200_1w_aligned[i]
        downtrend = ema50_1w_aligned[i] < ema200_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r4_aligned[i]
        short_breakout = close[i] < camarilla_s4_aligned[i]
        
        # Exit conditions: retreat to R3/S3 or opposite pivot level
        long_exit = close[i] < camarilla_r3_aligned[i] or close[i] < camarilla_s4_aligned[i]
        short_exit = close[i] > camarilla_s3_aligned[i] or close[i] > camarilla_r4_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and uptrend and volume_confirm and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and downtrend and volume_confirm and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "6h_1w_camarilla_breakout_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0