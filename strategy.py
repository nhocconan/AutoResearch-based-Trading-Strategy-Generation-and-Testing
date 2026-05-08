#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with daily EMA50 trend filter and volume spike
# Uses Donchian breakout on 4h timeframe, filtered by daily EMA50 trend and volume > 2x 20-period average.
# Designed to capture trends with strict entry conditions to limit trades and reduce fee drag.
# Target: 20-50 trades/year (80-200 total over 4 years). Works in bull/bear via trend filter.

name = "4h_Donchian20_DailyTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend using vectorized operations
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        # Vectorized EMA calculation
        alpha = 2 / (50 + 1)
        ema50_daily[0] = close_daily[0]
        for i in range(1, len(close_daily)):
            ema50_daily[i] = alpha * close_daily[i] + (1 - alpha) * ema50_daily[i-1]
    
    # Calculate 4h Donchian(20) using vectorized operations
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 4h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA50 to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close[i] > ema50_daily_aligned[i]
        price_below_ema = close[i] < ema50_daily_aligned[i]
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with trend and volume confirmation
            if close[i] > donchian_high[i] and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retrace to Donchian midpoint or trend fails or volume drops
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] < midpoint or not price_above_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retrace to Donchian midpoint or trend fails or volume drops
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if close[i] > midpoint or not price_below_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals