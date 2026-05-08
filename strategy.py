#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0, Bear Power < 0 (bullish), 12h EMA34 rising, volume > 1.3x average
# Short when Bull Power < 0, Bear Power > 0 (bearish), 12h EMA34 falling, volume > 1.3x average
# Uses 6h for Elder Ray calculation, 12h for trend filter to avoid whipsaws
# Targets 60-120 total trades over 4 years (15-30/year) for balanced freq and win rate

name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume spike: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        ema34_12h_val = ema34_12h_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Enter long: bullish Elder Ray (BP>0, BP<0), 12h uptrend, volume spike
            if bull_power_val > 0 and bear_power_val < 0 and ema34_12h_val > 0 and vol_spike_val:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish Elder Ray (BP<0, BP>0), 12h downtrend, volume spike
            elif bull_power_val < 0 and bear_power_val > 0 and ema34_12h_val < 0 and vol_spike_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish Elder Ray or 12h trend down
            if bull_power_val < 0 or bear_power_val > 0 or ema34_12h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish Elder Ray or 12h trend up
            if bull_power_val > 0 or bear_power_val < 0 or ema34_12h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals