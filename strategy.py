#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0, Bear Power < 0, price > EMA13, 1d EMA34 rising, volume > 1.5x avg
# Short when Bear Power < 0, Bull Power > 0, price < EMA13, 1d EMA34 falling, volume > 1.5x avg
# Uses Elder Ray for trend strength, EMA13 for entry filter, 1d EMA34 for trend filter, volume for confirmation
# Targets 12-37 trades per year (50-150 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation

name = "6h_ElderRay_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA13 for entry filter
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    ema13_for_power = ema13  # EMA13 used for Bull/Bear Power calculation
    bull_power = high - ema13_for_power
    bear_power = low - ema13_for_power
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days for volume MA and EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_val = ema13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        ema34_1d_val = ema34_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, price > EMA13, 1d uptrend, volume confirmation
            if (bull_power_val > 0 and bear_power_val < 0 and 
                close[i] > ema13_val and ema34_1d_val > 0 and vol_conf_val):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power > 0, price < EMA13, 1d downtrend, volume confirmation
            elif (bear_power_val < 0 and bull_power_val > 0 and 
                  close[i] < ema13_val and ema34_1d_val < 0 and vol_conf_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power >= 0 or price <= EMA13 or 1d trend down
            if (bear_power_val >= 0 or close[i] <= ema13_val or ema34_1d_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power <= 0 or price >= EMA13 or 1d trend up
            if (bull_power_val <= 0 or close[i] >= ema13_val or ema34_1d_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals