#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with daily trend filter and volume confirmation
# Long when Bull Power > 0 (close > EMA13) and Bear Power crosses above zero with volume spike.
# Short when Bear Power < 0 (close < EMA13) and Bull Power crosses below zero with volume spike.
# Uses 6h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Elder Ray measures bull/bear power relative to EMA, working in both trending and ranging markets.
# Daily trend filter (EMA34) ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation in the move.

name = "6h_ElderRay_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily EMA(13) for Elder Ray
    ema13_1d = pd.Series(daily_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume spike: current volume > 2.0 * 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power crosses above zero AND daily uptrend AND volume spike
            if (bull_power_val > 0 and bull_power_val <= bull_power[i-1] + 1e-9 and  # crossed above
                close[i] > ema34_1d_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power crosses below zero AND daily downtrend AND volume spike
            elif (bear_power_val < 0 and bear_power_val >= bear_power[i-1] - 1e-9 and  # crossed below
                  close[i] < ema34_1d_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power crosses above zero OR daily trend turns down
            if (bear_power_val > 0 and bear_power_val >= bear_power[i-1] - 1e-9) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power crosses below zero OR daily trend turns up
            if (bull_power_val < 0 and bull_power_val <= bull_power[i-1] + 1e-9) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals