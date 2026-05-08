#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume spike (>2x 20-period average)
# Uses Donchian breakout on daily timeframe for clear structure, filtered by weekly EMA50 trend and volume confirmation.
# Designed to capture trends in both bull and bear markets with strict entry conditions to limit trades.
# Target: 7-25 trades/year (30-100 total over 4 years).

name = "1d_Donchian20_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Calculate daily Donchian(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate daily volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA50 to daily timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close[i] > ema50_weekly_aligned[i]
        price_below_ema = close[i] < ema50_weekly_aligned[i]
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