#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation.
# Bull Power = High - EMA13(close); Bear Power = Low - EMA13(close).
# Long when Bull Power crosses above 0 AND price > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when Bear Power crosses below 0 AND price < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when opposite power crosses zero (Bear Power > 0 for long exit, Bull Power < 0 for short exit).
# Uses Elder Ray to measure bull/bear strength relative to trend, avoiding counter-trend trades.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Elder Ray: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA13 and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power crosses above 0, price above 1d EMA50, volume spike
            long_cond = (bull_power[i] > 0) and (bull_power[i-1] <= 0) and (close[i] > ema50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: Bear Power crosses below 0, price below 1d EMA50, volume spike
            short_cond = (bear_power[i] < 0) and (bear_power[i-1] >= 0) and (close[i] < ema50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power crosses above 0 (bulls losing strength)
            if bear_power[i] > 0 and bear_power[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power crosses below 0 (bears losing strength)
            if bull_power[i] < 0 and bull_power[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals