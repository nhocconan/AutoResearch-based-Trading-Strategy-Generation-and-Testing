#!/usr/bin/env python3
"""
Hypothesis: 1-day Williams %R with 1-week EMA trend filter and volume confirmation.
Trades reversals at Williams %R oversold/overbought levels (below 20/above 80) in the direction of the weekly EMA trend.
Uses volume spike to confirm institutional interest at key levels. Designed for low trade frequency
(20-50 trades/year) to minimize fee drag and work in both bull and bear markets by aligning with
higher timeframe trend and using mean-reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R for given high, low, close arrays."""
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Williams %R (14-period)
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: Williams %R below 20 (oversold) with uptrend bias
            if williams_r[i] < -20 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R above -80 (overbought) with downtrend bias
            elif williams_r[i] > -80 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50 or closes below weekly EMA
                if williams_r[i] > -50 or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50 or closes above weekly EMA
                if williams_r[i] < -50 or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_Reversal_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0