#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h EMA34 Trend + Daily Close Above/Below EMA + Volume Spike
# Uses daily EMA34 for trend bias, daily close above/below EMA34 for entry bias,
# and 12h volume spike (>2x 20-period average) for confirmation.
# Designed to capture trend continuation with confirmation. Target: 12-37 trades/year.

name = "12h_EMA34_DailyClose_Bias_VolumeSpike"
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
    
    # Get daily data for EMA trend and close price
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_daily = df_daily['close'].values
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate 12h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA34 to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current daily bar's close and EMA
        close_daily_current = np.nan
        if not np.isnan(ema34_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                close_daily_current = df_daily.iloc[idx_daily]['close']
        
        if np.isnan(close_daily_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close_daily_current > ema34_daily_aligned[i]
        price_below_ema = close_daily_current < ema34_daily_aligned[i]
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: follow daily EMA trend with volume spike
            if price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            elif price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA or volume drops
            if price_below_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA or volume drops
            if price_above_ema or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals