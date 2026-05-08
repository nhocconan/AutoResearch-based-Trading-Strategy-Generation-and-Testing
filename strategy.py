#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Trend Filter + Volume Spike
# Uses weekly trend direction for bias, Williams %R for mean reversion entries,
# and volume spike (>2x average) for confirmation. Designed to work in both bull and bear
# markets by following the weekly trend while buying dips in uptrends and selling rallies in downtrends.
# Target: 15-35 trades/year.

name = "6h_WilliamsR_1dTrend_VolumeSpike"
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
    
    # Get daily data for trend filter and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Get weekly data for Williams %R
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 14:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    willr_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 14:
        for i in range(13, len(close_weekly)):
            highest_high = np.max(high_weekly[i-13:i+1])
            lowest_low = np.min(low_weekly[i-13:i+1])
            if highest_high > lowest_low:
                willr_weekly[i] = -100 * (highest_high - close_weekly[i]) / (highest_high - lowest_low)
            else:
                willr_weekly[i] = -50  # neutral when no range
    
    # Align daily and weekly indicators to 6h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    willr_weekly_aligned = align_htf_to_ltf(prices, df_weekly, willr_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i]) or
            np.isnan(willr_weekly_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 6h volume > 2x 20-period average of daily volume
        vol_spike = volume[i] > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Williams %R oversold/overbought with weekly trend and volume spike
            # Long when weekly trend up, Williams %R oversold (< -80), and volume spike
            long_condition = (
                close[i] > ema34_daily_aligned[i] and   # price above daily EMA34 (bullish bias)
                willr_weekly_aligned[i] < -80 and       # Williams %R oversold
                vol_spike                               # volume spike for entry
            )
            
            # Short when weekly trend down, Williams %R overbought (> -20), and volume spike
            short_condition = (
                close[i] < ema34_daily_aligned[i] and   # price below daily EMA34 (bearish bias)
                willr_weekly_aligned[i] > -20 and       # Williams %R overbought
                vol_spike                               # volume spike for entry
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 or trend changes
            if willr_weekly_aligned[i] > -50 or close[i] < ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 or trend changes
            if willr_weekly_aligned[i] < -50 or close[i] > ema34_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals