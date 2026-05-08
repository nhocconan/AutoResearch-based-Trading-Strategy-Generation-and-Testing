# 2025-06-24: 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla pivot level breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R1 (1.049*close - 0.049*low) AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S1 (1.049*close - 0.049*high) AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Camarilla H-L range (between H3 and L3).
# This strategy targets 50-150 total trades over 4 years (12-37/year) by using tight breakout conditions on 12h timeframe.
# Camarilla levels provide mathematically derived support/resistance. The 1d EMA34 ensures higher timeframe trend alignment.
# Volume spike confirms institutional participation, reducing false breakouts.
# Works in both bull and bear markets: trend filter adapts to direction, breakouts capture momentum in any regime.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (using previous day's OHLC)
    # Camarilla formulas: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.0*(high-low)
    # H2 = close + 0.5*(high-low)
    # H1 = close + 0.25*(high-low)
    # L1 = close - 0.25*(high-low)
    # L2 = close - 0.5*(high-low)
    # L3 = close - 1.0*(high-low)
    # L4 = close - 1.5*(high-low)
    # We use R1 = H1 and S1 = L1 for breakout entries
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    camarilla_H1 = prev_close + 0.25 * (prev_high - prev_low)  # R1
    camarilla_L1 = prev_close - 0.25 * (prev_high - prev_low)  # S1
    camarilla_H3 = prev_close + 1.0 * (prev_high - prev_low)   # H3 for exit
    camarilla_L3 = prev_close - 1.0 * (prev_high - prev_low)   # L3 for exit
    
    # Align Camarilla levels to 12h timeframe
    camarilla_H1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H1)
    camarilla_L1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L1)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_H1_aligned[i]) or np.isnan(camarilla_L1_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R1 (H1), 1d EMA34 rising, volume filter
            long_cond = (close[i] > camarilla_H1_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below Camarilla S1 (L1), 1d EMA34 falling, volume filter
            short_cond = (close[i] < camarilla_L1_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla L3
            if close[i] < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla H3
            if close[i] > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals