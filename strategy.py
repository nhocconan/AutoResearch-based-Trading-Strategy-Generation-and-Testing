#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly EMA50 Trend + Daily Price Action + Volume Spike
# Uses weekly EMA50 for trend bias, daily price closing above/below weekly EMA for entry,
# and volume spike (>2x 20-day average) for confirmation. Designed to capture trend
# continuation in both bull and bear markets by following the weekly trend while avoiding
# false signals with volume confirmation. Target: 12-37 trades/year.

name = "12h_WeeklyEMA50_DailyPriceAction_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Get daily data for price action and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Calculate daily volume average for volume spike
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly and daily indicators to 12h timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
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
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current daily bar's volume and price action
        vol_spike = False
        price_above_weekly_ema = False
        price_below_weekly_ema = False
        
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's data
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_12h_current = volume[i]
                vol_spike = vol_12h_current > 2.0 * vol_avg_20_daily_aligned[i]
                
                # Check if current 12h price is above/below weekly EMA
                price_above_weekly_ema = close[i] > ema50_weekly_aligned[i]
                price_below_weekly_ema = close[i] < ema50_weekly_aligned[i]
        
        if position == 0:
            # Look for entry: follow weekly EMA trend with volume spike
            if price_above_weekly_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif price_below_weekly_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below weekly EMA or volume drops
            if price_below_weekly_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above weekly EMA or volume drops
            if price_above_weekly_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals