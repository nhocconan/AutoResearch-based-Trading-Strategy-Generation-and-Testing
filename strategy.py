#!/usr/bin/env python3
name = "6h_ElderRay_BullBearPower_1dTrend_1wFilter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 13-period EMA for Elder Ray (standard)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Weekly trend filter: 21-period EMA
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate Elder Ray components
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    position_size = 0.25
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # sufficient warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(ema21_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend alignment: weekly EMA21 slope (using prior bar)
        weekly_uptrend = ema21_1w_aligned[i] > ema21_1w_aligned[i-1]
        weekly_downtrend = ema21_1w_aligned[i] < ema21_1w_aligned[i-1]
        
        if position == 0:
            # Long: Bull power positive + weekly uptrend + volume spike
            if bull_power[i] > 0 and weekly_uptrend and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Bear power negative + weekly downtrend + volume spike
            elif bear_power[i] < 0 and weekly_downtrend and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit when power fades or trend reverses
            if position == 1:
                # Exit long: bull power turns negative OR weekly trend turns down
                if bull_power[i] <= 0 or not weekly_uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: bear power turns positive OR weekly trend turns up
                if bear_power[i] >= 0 or not weekly_downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals