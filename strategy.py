#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (bull/bear power) with 1d EMA34 trend filter and volume confirmation.
# Long when bull power > 0, price > EMA34(1d), and volume > 1.5x 20-period average.
# Short when bear power < 0, price < EMA34(1d), and volume > 1.5x 20-period average.
# Exit when bull/bear power crosses zero or price crosses EMA34.
# Elder Ray measures bull/bear power relative to EMA13. EMA34 filters trend direction.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull power: high - EMA13
    bear_power = low - ema13   # Bear power: low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bull power > 0, price > EMA34, volume filter
            long_cond = (bull_power[i] > 0) and (close[i] > ema_34_aligned[i]) and volume_filter[i]
            # Short conditions: bear power < 0, price < EMA34, volume filter
            short_cond = (bear_power[i] < 0) and (close[i] < ema_34_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bull power <= 0 or price <= EMA34
            if (bull_power[i] <= 0) or (close[i] <= ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bear power >= 0 or price >= EMA34
            if (bear_power[i] >= 0) or (close[i] >= ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals