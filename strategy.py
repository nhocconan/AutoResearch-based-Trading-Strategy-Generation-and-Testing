#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Trade 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation.
- Trend filter: price > 4h EMA50 = bullish, price < 4h EMA50 = bearish.
- In bullish 4h trend: buy breakouts above R1, sell breakdowns below S1.
- In bearish 4h trend: sell breakdowns below S1, buy breakouts above R1 (continuation logic).
- Volume confirmation: require volume > 2.0x 20-period average to avoid false breakouts.
- Session filter: trade only 08:00-20:00 UTC to avoid low-liquidity hours.
- Exit on trend reversal or mean reversion to pivot.
- Position size: 0.20. Target: 60-150 total trades over 4 years = 15-37/year.
- Uses 4h for signal direction, 1h only for entry timing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for 08:00-20:00 UTC filter
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 4h data for HTF trend filter and Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (using previous 4h bar's OHLC)
    prev_close = np.roll(df_4h['close'].values, 1)
    prev_high = np.roll(df_4h['high'].values, 1)
    prev_low = np.roll(df_4h['low'].values, 1)
    prev_close[0] = df_4h['close'].values[0]
    prev_high[0] = df_4h['high'].values[0]
    prev_low[0] = df_4h['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade 08:00-20:00 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Determine 4h HTF trend using EMA50
        htf_4h_bullish = close[i] > ema_50_4h_aligned[i]
        htf_4h_bearish = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Breakout logic: trade in direction of 4h trend with volume confirmation
            long_setup = (close[i] > r1_aligned[i]) and htf_4h_bullish and volume_spike[i]
            short_setup = (close[i] < s1_aligned[i]) and htf_4h_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = (not htf_4h_bullish) or (close[i] < pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit on trend reversal or mean reversion to pivot
            exit_signal = htf_4h_bullish or (close[i] > pivot_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0