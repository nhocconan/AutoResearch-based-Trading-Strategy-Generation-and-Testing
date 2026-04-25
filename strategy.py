#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1w EMA50 trend filter and volume spike confirmation.
- In weekly uptrend (price > 1w EMA50): buy breakouts above R1, sell breakdowns below S1.
- In weekly downtrend (price < 1w EMA50): fade extremes at R1/S1 with mean reversion.
- Volume confirmation: require volume > 2.0x 50-period average to avoid false breakouts.
- Position size: 0.25. Target: 75-150 total trades over 4 years = 19-38/year.
- Works in both bull and bear: weekly trend filter adapts to market regime, volume filters noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = df_1d['close'].values[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike confirmation: volume > 2.0x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1w HTF trend (bullish = price above 1w EMA50)
        htf_1w_bullish = close[i] > ema_50_1w_aligned[i]
        htf_1w_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Determine if we are in trending or ranging market based on distance from EMA
        ema_distance = abs(close[i] - ema_50_1w_aligned[i]) / ema_50_1w_aligned[i]
        trending_market = ema_distance > 0.03  # >3% away from EMA = trending
        ranging_market = ema_distance <= 0.03   # <=3% away from EMA = ranging
        
        if position == 0:
            if trending_market:
                # Trending market: trade breakout continuation
                long_setup = (close[i] > r1_aligned[i]) and htf_1w_bullish and volume_spike[i]
                short_setup = (close[i] < s1_aligned[i]) and htf_1w_bearish and volume_spike[i]
            else:
                # Ranging market: trade mean reversion at extremes
                long_setup = (close[i] < s1_aligned[i]) and (close[i] > s3_aligned[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] > r1_aligned[i]) and (close[i] < r3_aligned[i]) and volume_spike[i]  # Overbought rejection
            
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
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of S1
                exit_signal = (not htf_1w_bullish) or (close[i] < s1_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of R1
                exit_signal = (close[i] > pivot_aligned[i]) or (close[i] > r1_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of R1
                exit_signal = htf_1w_bullish or (close[i] > r1_aligned[i])
            else:
                # In ranging market: exit on mean reversion to pivot or touch of S1
                exit_signal = (close[i] < pivot_aligned[i]) or (close[i] < s1_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0