#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams %R Extreme Reversal with 12-hour EMA Trend Filter and Volume Confirmation.
Trades reversals at Williams %R extreme levels (<10 for oversold, >90 for overbought) in the direction of the 12-hour EMA trend.
Uses volume spike to confirm institutional interest at turning points. Designed for low trade frequency
(15-30 trades/year) to minimize fee dust and work in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period=14):
    """Calculate Williams %R indicator."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.fillna(0).values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend filter and Williams %R calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12-hour EMA for trend filter (34-period)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12-hour Williams %R (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    wr_14 = williams_r(high_12h, low_12h, close_12h_arr, 14)
    wr_14_aligned = align_htf_to_ltf(prices, df_12h, wr_14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(wr_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R oversold (< -90) with uptrend bias
            if wr_14_aligned[i] < -90 and close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -10) with downtrend bias
            elif wr_14_aligned[i] > -10 and close[i] < ema_34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or price closes below 12h EMA
                if wr_14_aligned[i] > -50 or close[i] < ema_34_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or price closes above 12h EMA
                if wr_14_aligned[i] < -50 or close[i] > ema_34_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0