#!/usr/bin/env python3
"""
6h Elder Power + Weekly Pivot Direction + Volume Spike
Hypothesis: Elder Ray Bull/Bear Power from daily chart confirms institutional buying/selling pressure,
combined with weekly Camarilla pivot direction for structural bias and volume spike for momentum.
Works in both bull and bear markets by focusing on power imbalance rather than pure trend.
Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing: 0.25.
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
    
    # Get 1d data for Elder Ray Power (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Elder Ray Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    daily_close = df_1d['close'].values
    ema13 = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 1w data for Camarilla pivot direction (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla H3/L3 for directional bias
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    camarilla_h3_weekly = weekly_close + 1.1 * (weekly_high - weekly_low) / 4
    camarilla_l3_weekly = weekly_close - 1.1 * (weekly_high - weekly_low) / 4
    camarilla_h3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_weekly)
    camarilla_l3_weekly_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_weekly)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA13, ATR, and volume MA to propagate
    start_idx = max(13, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(camarilla_h3_weekly_aligned[i]) or 
            np.isnan(camarilla_l3_weekly_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        h3_weekly = camarilla_h3_weekly_aligned[i]
        l3_weekly = camarilla_l3_weekly_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) AND price above weekly H3 AND volume spike
            long_condition = (bull_power_val > 0) and (curr_close > h3_weekly) and volume_spike
            # Short: Bear Power < 0 (selling pressure) AND price below weekly L3 AND volume spike
            short_condition = (bear_power_val < 0) and (curr_close < l3_weekly) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or Bull Power turns negative
            if curr_close <= entry_price - 2.0 * atr_val or bull_power_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or Bear Power turns positive
            if curr_close >= entry_price + 2.0 * atr_val or bear_power_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderPower_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0