#!/usr/bin/env python3
"""
6h_WeeklyPivot_Camarilla_Breakout_Trend_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 6h timeframe filtered by weekly pivot bias (from 1w data) and 1d EMA50 trend. Uses volume confirmation and ATR trailing stop.
Weekly pivot bias ensures we only trade in alignment with the higher-timeframe structure, improving performance in both bull and bear markets by avoiding counter-trend whipsaws.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Position size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot bias (Camarilla R4/S4 as bias levels)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly Camarilla levels from previous 1w bar
    prev_weekly_close = df_1w['close'].shift(1).values
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    weekly_r4 = prev_weekly_close + (prev_weekly_high - prev_weekly_low) * 1.1 / 2  # R4
    weekly_s4 = prev_weekly_close - (prev_weekly_high - prev_weekly_low) * 1.1 / 2  # S4
    
    # Get 1d data for intraday Camarilla levels (R1/S1 for entry)
    prev_daily_close = df_1d['close'].shift(1).values
    prev_daily_high = df_1d['high'].shift(1).values
    prev_daily_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 12
    camarilla_s1 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 12
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: 2.0x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (50), weekly pivot, volume median (30), 6h ATR (14)
    start_idx = max(50, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(weekly_r4_aligned[i]) or 
            np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        weekly_r4_val = weekly_r4_aligned[i]
        weekly_s4_val = weekly_s4_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: price > weekly R4 (bullish bias), break above R1, uptrend (close > EMA50), volume spike
            long_signal = (close_val > weekly_r4_val) and \
                          (high_val > camarilla_r1_val) and \
                          (close_val > ema_50_1d_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: price < weekly S4 (bearish bias), break below S1, downtrend (close < EMA50), volume spike
            short_signal = (close_val < weekly_s4_val) and \
                           (low_val < camarilla_s1_val) and \
                           (close_val < ema_50_1d_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Camarilla_Breakout_Trend_v1"
timeframe = "6h"
leverage = 1.0