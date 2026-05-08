#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot level breakout with 12-hour EMA trend filter and volume confirmation
# Camarilla pivot levels (R1, S1) act as strong support/resistance in sideways and trending markets
# Long when price breaks above R1 with 12h EMA50 uptrend and volume > 1.5x average
# Short when price breaks below S1 with 12h EMA50 downtrend and volume > 1.5x average
# Exit when price returns to pivot point (PP) or reverses
# Targets 25-35 trades/year to minimize fee decay while capturing sustained trends

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Volume"
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
    
    # Get daily data for Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h_50 = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if i < 50:
            ema_12h_50[i] = np.mean(close_12h[max(0, i-49):i+1]) if i >= 0 else close_12h[i]
        else:
            ema_12h_50[i] = np.mean(close_12h[i-49:i+1])
    
    # Calculate daily volume average for volume confirmation
    daily_volume = df_daily['volume'].values
    vol_ma_20 = np.full_like(daily_volume, np.nan)
    for i in range(len(daily_volume)):
        if i < 20:
            vol_ma_20[i] = np.mean(daily_volume[max(0, i-19):i+1]) if i >= 0 else daily_volume[i]
        else:
            vol_ma_20[i] = np.mean(daily_volume[i-19:i+1])
    
    # Align 12h EMA and daily volume to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find the most recent completed daily bar for Camarilla calculation
        idx_daily = len(df_daily) - 1
        while idx_daily >= 0 and df_daily.iloc[idx_daily]['open_time'] > prices.iloc[i]['open_time']:
            idx_daily -= 1
        
        if idx_daily < 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels from previous day
        prev_high = df_daily.iloc[idx_daily]['high']
        prev_low = df_daily.iloc[idx_daily]['low']
        prev_close = df_daily.iloc[idx_daily]['close']
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r1 = pivot + (range_val * 1.1 / 12)
        s1 = pivot - (range_val * 1.1 / 12)
        
        # Volume filter: current daily volume > 1.5x 20-day average
        vol_daily_current = df_daily.iloc[idx_daily]['volume']
        vol_filter = vol_daily_current > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: 12h EMA50 direction
        ema_now = ema_12h_50_aligned[i]
        ema_prev = ema_12h_50_aligned[i-1] if i > 0 else ema_now
        ema_uptrend = ema_now > ema_prev
        ema_downtrend = ema_now < ema_prev
        
        if position == 0:
            # Look for Camarilla breakout with volume and trend confirmation
            if high[i] > r1 and vol_filter and ema_uptrend:
                signals[i] = 0.25
                position = 1
            elif low[i] < s1 and vol_filter and ema_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to pivot or trend reverses
            if low[i] <= pivot or not ema_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to pivot or trend reverses
            if high[i] >= pivot or not ema_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals