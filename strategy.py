#!/usr/bin/env python3
# 6H_ElderRay_BullBearPower_WeeklyTrend
# Hypothesis: Use Elder Ray (Bull/Bear Power) on 6h with weekly trend filter and volume confirmation.
# Long when: Bear Power < 0 (bears weak) AND weekly uptrend AND volume > 1.5x average.
# Short when: Bull Power > 0 (bulls weak) AND weekly downtrend AND volume > 1.5x average.
# Works in bull/bear by following weekly trend and using Elder Ray to detect weakness in opposing force.
# Target: 12-30 trades/year per symbol.

name = "6H_ElderRay_BullBearPower_WeeklyTrend"
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
    
    # 6h indicators
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # EMA13 for Elder Ray
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 6h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + Bear Power < 0 (bears weak) + volume
            if weekly_up and bear_power[i] < 0 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + Bull Power > 0 (bulls weak) + volume
            elif weekly_down and bull_power[i] > 0 and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: weekly trend changes or Bull Power becomes negative (bulls weak)
            if not weekly_up or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: weekly trend changes or Bear Power becomes positive (bears strong)
            if not weekly_down or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals