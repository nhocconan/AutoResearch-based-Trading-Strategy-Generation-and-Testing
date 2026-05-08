#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and 6h volume confirmation.
# Long when Bull Power > 0 (close > EMA13), Bear Power < 0 (low < EMA13), 1d EMA34 up, and 6h volume > 1.5x 20-period average.
# Short when Bull Power < 0, Bear Power > 0, 1d EMA34 down, and volume spike.
# Exit when Bull Power and Bear Power converge (|Bull Power - Bear Power| < 0.1 * ATR) or trend reverses.
# Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) for low fee drift.

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
    
    # 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # 6h ATR for exit condition
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_slope = ema34_1d - np.roll(ema34_1d, 1)
    ema34_1d_slope[0] = 0
    
    # Align 1d EMA34 slope to 6h timeframe
    ema34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_1d_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power > 0, 1d EMA34 up, volume spike
            long_cond = (bull_power[i] > 0) and (bear_power[i] > 0) and (ema34_1d_slope_aligned[i] > 0) and volume_filter[i]
            # Short conditions: Bull Power < 0, Bear Power < 0, 1d EMA34 down, volume spike
            short_cond = (bull_power[i] < 0) and (bear_power[i] < 0) and (ema34_1d_slope_aligned[i] < 0) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: power convergence or trend reversal
            power_diff = np.abs(bull_power[i] - bear_power[i])
            exit_cond = (power_diff < 0.1 * atr[i]) or (ema34_1d_slope_aligned[i] < 0)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: power convergence or trend reversal
            power_diff = np.abs(bull_power[i] - bear_power[i])
            exit_cond = (power_diff < 0.1 * atr[i]) or (ema34_1d_slope_aligned[i] > 0)
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals