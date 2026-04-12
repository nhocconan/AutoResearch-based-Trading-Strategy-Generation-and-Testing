#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Volume_v1
Hypothesis: Weekly and daily Camarilla levels provide strong support/resistance on 12h timeframe.
Breakouts above resistance with volume confirmation signal continuation; bounces from support with
volume and RSI signal reversals. Works in trending and ranging markets by adapting to price action
relative to institutional pivot levels. Target: 15-25 trades per year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Volume_v1"
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
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for entries
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === WEEKLY CAMARILLA LEVELS (trend filter) ===
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    
    weekly_pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Weekly resistance and support
    weekly_r4 = weekly_pivot + (weekly_range * 1.1 / 2)  # Strong resistance
    weekly_s4 = weekly_pivot - (weekly_range * 1.1 / 2)  # Strong support
    
    weekly_r4_12h = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_12h = align_htf_to_ltf(prices, df_1w, weekly_s4)
    
    # === DAILY CAMARILLA LEVELS (entry levels) ===
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_range = prev_daily_high - prev_daily_low
    
    # Daily key levels
    daily_r3 = daily_pivot + (daily_range * 1.1 / 4)
    daily_r4 = daily_pivot + (daily_range * 1.1 / 2)
    daily_s3 = daily_pivot - (daily_range * 1.1 / 4)
    daily_s4 = daily_pivot - (daily_range * 1.1 / 2)
    
    daily_r3_12h = align_htf_to_ltf(prices, df_1d, daily_r3)
    daily_r4_12h = align_htf_to_ltf(prices, df_1d, daily_r4)
    daily_s3_12h = align_htf_to_ltf(prices, df_1d, daily_s3)
    daily_s4_12h = align_htf_to_ltf(prices, df_1d, daily_s4)
    
    # === VOLUME SPIKE (2x 20-period average on 12h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 2.0)
    
    # === RSI ON DAILY (filter for overextended moves) ===
    delta = pd.Series(daily_close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    daily_rsi = 100 - (100 / (1 + rs))
    daily_rsi_12h = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(weekly_r4_12h[i]) or np.isnan(weekly_s4_12h[i]) or
            np.isnan(daily_r3_12h[i]) or np.isnan(daily_r4_12h[i]) or
            np.isnan(daily_s3_12h[i]) or np.isnan(daily_s4_12h[i]) or
            np.isnan(daily_rsi_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine market regime based on weekly levels
        above_weekly_r4 = close[i] > weekly_r4_12h[i]
        below_weekly_s4 = close[i] < weekly_s4_12h[i]
        
        # Price proximity to daily levels (within 0.2%)
        near_daily_r3 = abs(high[i] - daily_r3_12h[i]) / daily_r3_12h[i] < 0.002
        near_daily_r4 = abs(high[i] - daily_r4_12h[i]) / daily_r4_12h[i] < 0.002
        near_daily_s3 = abs(low[i] - daily_s3_12h[i]) / daily_s3_12h[i] < 0.002
        near_daily_s4 = abs(low[i] - daily_s4_12h[i]) / daily_s4_12h[i] < 0.002
        
        # Entry logic adapts to weekly trend
        if above_weekly_r4:
            # In weekly uptrend: look for long bounces from support
            long_entry = (near_daily_s3 or near_daily_s4) and daily_rsi_12h[i] < 40 and vol_spike[i]
            short_entry = False  # Avoid shorts in strong uptrend
        elif below_weekly_s4:
            # In weekly downtrend: look for short bounces from resistance
            short_entry = (near_daily_r3 or near_daily_r4) and daily_rsi_12h[i] > 60 and vol_spike[i]
            long_entry = False   # Avoid longs in strong downtrend
        else:
            # In weekly range: trade both directions from key levels
            long_entry = (near_daily_s3 or near_daily_s4) and daily_rsi_12h[i] < 35 and vol_spike[i]
            short_entry = (near_daily_r3 or near_daily_r4) and daily_rsi_12h[i] > 65 and vol_spike[i]
        
        # Exit when price reaches opposite daily level or weekly extreme
        long_exit = close[i] >= daily_r3_12h[i] or close[i] >= weekly_r4_12h[i]
        short_exit = close[i] <= daily_r3_12h[i] or close[i] <= weekly_s4_12h[i]
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals