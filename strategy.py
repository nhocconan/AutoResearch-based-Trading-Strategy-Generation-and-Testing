#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
Hypothesis: Donchian breakouts from the 6h chart capture momentum, filtered by 
weekly pivot direction (from 1w timeframe) to ensure alignment with higher-timeframe 
trend, and volume spike confirmation to avoid false breakouts. Works in both bull 
and bear markets by only taking breakouts in the direction of the weekly trend.
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
    
    # Get 1w data for weekly pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Weekly trend direction: price above weekly pivot = uptrend, below = downtrend
    weekly_trend_up = weekly_close > weekly_pivot
    weekly_trend_down = weekly_close < weekly_pivot
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Calculate Donchian channels (20-period) on 6h data
    donchian_period = 20
    if len(close) >= donchian_period:
        donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
        donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate ATR(14) for stoploss
    atr_period = 14
    if len(close) >= atr_period:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period, min_periods=atr_period).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_period = 20
    vol_ma = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - vol_ma_period + 1)
        vol_ma[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian, ATR, volume MA to propagate
    start_idx = max(donchian_period, atr_period, vol_ma_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        weekly_trend_up_val = weekly_trend_up_aligned[i]
        weekly_trend_down_val = weekly_trend_down_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above Donchian high AND weekly uptrend AND volume spike
            long_condition = (curr_close > dc_high) and (weekly_trend_up_val > 0.5) and volume_spike
            # Short: price breaks below Donchian low AND weekly downtrend AND volume spike
            short_condition = (curr_close < dc_low) and (weekly_trend_down_val > 0.5) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price breaks below Donchian low
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price breaks above Donchian high
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0