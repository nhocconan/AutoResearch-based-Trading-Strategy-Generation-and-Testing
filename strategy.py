#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R Mean Reversion with 1-day Trend Filter and Volume Confirmation.
Long when Williams %R crosses above -80 (oversold) during 1-day uptrend with volume spike.
Short when Williams %R crosses below -20 (overbought) during 1-day downtrend with volume spike.
Exit when Williams %R returns to -50 (mean) or trend reverses.
Williams %R is effective in ranging markets and captures reversals in trends.
Designed for moderate trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or (highest_high[i] - lowest_low[i]) == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold) + 1d uptrend + volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) + 1d downtrend + volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to -50 (mean) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R >= -50 or 1d trend turns down
                if williams_r[i] >= -50 or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R <= -50 or 1d trend turns up
                if williams_r[i] <= -50 or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_MeanReversion_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0