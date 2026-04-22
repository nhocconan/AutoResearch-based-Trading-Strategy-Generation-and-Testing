#!/usr/bin/env python3

"""
Hypothesis: 4-hour Williams %R Reversal with 1-week EMA trend filter and volume confirmation.
Trades reversals at extreme Williams %R levels (>80 for oversold, <20 for overbought) in the direction of the weekly EMA trend.
Uses volume spike to confirm institutional interest at key levels. Designed for low trade frequency (20-50 trades/year) to minimize fee drag
and work in both bull and bear markets by aligning with higher timeframe trend and using mean-reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (30-period)
    close_1w = df_1w['close'].values
    ema_30_1w = pd.Series(close_1w).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # Williams %R (14-period)
    wr = calculate_williams_r(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_30_1w_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R oversold (< -80) with uptrend bias
            if wr[i] < -80 and close[i] > ema_30_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with downtrend bias
            elif wr[i] > -20 and close[i] < ema_30_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or closes below weekly EMA
                if wr[i] > -50 or close[i] < ema_30_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or closes above weekly EMA
                if wr[i] < -50 or close[i] > ema_30_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1wEMA30_Volume"
timeframe = "4h"
leverage = 1.0