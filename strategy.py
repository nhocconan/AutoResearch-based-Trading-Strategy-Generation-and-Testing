#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian channel breakout with weekly trend filter and volume confirmation.
Long when price breaks above 12h Donchian upper (20) with weekly EMA40 rising and volume spike.
Short when price breaks below 12h Donchian lower (20) with weekly EMA40 falling and volume spike.
Exit when price returns to 12h Donchian midpoint.
Designed for low trade frequency (<30/year) by requiring multiple confirmations.
Works in bull markets via breakouts and bear markets via short breakdowns.
"""

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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Weekly EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # 12h Donchian channel (20-period)
    donch_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    midpoint = np.full(n, np.nan)
    
    for i in range(donch_len - 1, n):
        window_high = high[i - donch_len + 1:i + 1]
        window_low = low[i - donch_len + 1:i + 1]
        upper[i] = np.max(window_high)
        lower[i] = np.min(window_low)
        midpoint[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for weekly EMA
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema40_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_30[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with weekly EMA40 rising and volume spike
            if (close[i] > upper[i] and 
                ema40_1w_aligned[i] > ema40_1w_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with weekly EMA40 falling and volume spike
            elif (close[i] < lower[i] and 
                  ema40_1w_aligned[i] < ema40_1w_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < midpoint[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint
                if close[i] > midpoint[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_WeeklyEMA40_Volume"
timeframe = "12h"
leverage = 1.0