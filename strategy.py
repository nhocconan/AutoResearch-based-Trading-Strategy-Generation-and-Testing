#!/usr/bin/env python3
name = "6h_1d_WeeklyPivot_DonchianBreakout_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot from previous week
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    weekly_range = prev_week_high - prev_week_low
    
    # Weekly support/resistance levels (similar to Camarilla but simpler)
    weekly_s1 = prev_week_close - (weekly_range * 1.08 / 2)
    weekly_r1 = prev_week_close + (weekly_range * 1.08 / 2)
    weekly_s2 = prev_week_close - (weekly_range * 1.16 / 2)
    weekly_r2 = prev_week_close + (weekly_range * 1.16 / 2)
    weekly_s3 = prev_week_close - (weekly_range * 1.26 / 4)
    weekly_r3 = prev_week_close + (weekly_range * 1.26 / 4)
    
    # Align weekly levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1w, weekly_s3)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1w, weekly_r3)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channel (20 periods) on 6h data
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume spike detection: 24-period average (4 days of 6h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, donchian_period, 24)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            weekly_uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and weekly downtrend
            elif close[i] < donchian_low[i] and vol_condition and not weekly_uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to weekly pivot or volume drops
            if close[i] < weekly_pivot_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to weekly pivot or volume drops
            if close[i] > weekly_pivot_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Donchian breakout with weekly pivot direction and volume confirmation
# - Weekly pivot acts as major support/resistance level from higher timeframe
# - Donchian(20) breakout captures medium-term momentum
# - Volume spike (2.0x average) confirms institutional participation
# - Weekly trend filter (daily EMA34) ensures alignment with higher timeframe momentum
# - Works in both bull (buy Donchian breaks in weekly uptrend) and bear (sell Donchian breaks in weekly downtrend)
# - Exit when price returns to weekly pivot or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual weekly pivot levels for institutional relevance
# - Designed to work in BOTH bull and bear markets via trend filter