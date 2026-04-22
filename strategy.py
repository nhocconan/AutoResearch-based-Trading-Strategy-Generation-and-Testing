#!/usr/bin/env python3

"""
Hypothesis: 12-hour Williams %R Mean Reversion with 1-day EMA trend filter and volume confirmation.
Trades reversals at extreme Williams %R levels (oversold < -80, overbought > -20) in the direction of the daily EMA trend.
Uses volume spike to confirm institutional interest at key levels. Designed for low trade frequency
(12-37 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with
higher timeframe trend and using mean-reversion at extreme momentum levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r.fillna(0).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Williams %R calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    williams_r = calculate_williams_r(high_1d, low_1d, close_1d_arr, 14)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R oversold (< -80) with uptrend bias
            if williams_r_aligned[i] < -80 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with downtrend bias
            elif williams_r_aligned[i] > -20 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or price closes below daily EMA
                if williams_r_aligned[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or price closes above daily EMA
                if williams_r_aligned[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_MeanReversion_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0