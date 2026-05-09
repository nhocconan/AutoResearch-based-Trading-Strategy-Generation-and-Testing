#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyPivot_R4S4_Breakout_WeeklyTrend_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot and trend
    df_wk = get_htf_data(prices, '1w')
    
    if len(df_wk) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot and levels from previous week's OHLC
    close_wk = df_wk['close'].values
    high_wk = df_wk['high'].values
    low_wk = df_wk['low'].values
    
    prev_high_wk = np.roll(high_wk, 1)
    prev_low_wk = np.roll(low_wk, 1)
    prev_close_wk = np.roll(close_wk, 1)
    prev_high_wk[0] = np.nan
    prev_low_wk[0] = np.nan
    prev_close_wk[0] = np.nan
    
    prev_weekly_range = prev_high_wk - prev_low_wk
    pivot = (prev_high_wk + prev_low_wk + prev_close_wk) / 3
    r4 = pivot + 1.1 * prev_weekly_range * 1.1
    s4 = pivot - 1.1 * prev_weekly_range * 1.1
    
    # Align weekly pivot levels to daily
    r4_daily = align_htf_to_ltf(prices, df_wk, r4)
    s4_daily = align_htf_to_ltf(prices, df_wk, s4)
    
    # Weekly EMA34 for trend filter
    ema34_wk = pd.Series(df_wk['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily = align_htf_to_ltf(prices, df_wk, ema34_wk)
    
    # Volume spike detection (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_daily[i]) or np.isnan(s4_daily[i]) or np.isnan(ema34_daily[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.5
        
        if position == 0:
            # Long: Break above weekly R4 with uptrend and volume spike
            if close[i] > r4_daily[i] and close[i] > ema34_daily[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly S4 with downtrend and volume spike
            elif close[i] < s4_daily[i] and close[i] < ema34_daily[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly S4 OR trend turns down
            if close[i] < s4_daily[i] or close[i] < ema34_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly R4 OR trend turns up
            if close[i] > r4_daily[i] or close[i] > ema34_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals