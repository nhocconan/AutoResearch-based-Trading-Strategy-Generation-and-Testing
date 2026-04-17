#!/usr/bin/env python3
"""
4h_SR_Touch_Trend_Breakout_v1
4-hour strategy: enter when price touches weekly support/resistance (from daily) with volume confirmation and daily trend alignment.
Exit on opposite touch or trend reversal.
Target: 80-120 total trades over 4 years (20-30/year). Works in bull/bear via S/R + trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Support/Resistance: 20-bar high/low ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily resistance: highest high of last 20 days
    dh_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Daily support: lowest low of last 20 days
    dl_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    dh_20_aligned = align_htf_to_ltf(prices, df_1d, dh_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1d, dl_20)
    
    # === Daily trend: EMA34 slope ===
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_slope = ema_34 - np.roll(ema_34, 1)  # daily change
    ema_34_slope[0] = 0
    ema_34_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_slope)
    
    # === Volume confirmation: current daily volume > 1.8x 20-day average ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_20_aligned[i]) or 
            np.isnan(dl_20_aligned[i]) or 
            np.isnan(ema_34_slope_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(vol_1d_current[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = vol_1d_current[i] > 1.8 * vol_ma_20_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price touches or breaks above daily resistance with volume + uptrend
            if (high[i] >= dh_20_aligned[i] and vol_confirmed and ema_34_slope_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price touches or breaks below daily support with volume + downtrend
            elif (low[i] <= dl_20_aligned[i] and vol_confirmed and ema_34_slope_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price touches or breaks below daily support OR trend turns down
            if (low[i] <= dl_20_aligned[i] or ema_34_slope_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches or breaks above daily resistance OR trend turns up
            if (high[i] >= dh_20_aligned[i] or ema_34_slope_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_SR_Touch_Trend_Breakout_v1"
timeframe = "4h"
leverage = 1.0