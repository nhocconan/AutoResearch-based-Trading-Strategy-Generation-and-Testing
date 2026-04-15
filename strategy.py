#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe trend
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly Supertrend for trend filter
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.roll(weekly_close, 1))
    tr3 = np.abs(weekly_low - np.roll(weekly_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Calculate ATR
    atr = np.zeros_like(weekly_close)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(weekly_close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate basic upper and lower bands
    hl2 = (weekly_high + weekly_low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize Supertrend
    supertrend = np.zeros_like(weekly_close)
    direction = np.ones_like(weekly_close)  # 1 for uptrend, -1 for downtrend
    
    # First valid value
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(weekly_close)):
        if close[i] <= supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = 1
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, weekly, supertrend)
    direction_aligned = align_htf_to_ltf(prices, weekly, direction)
    
    # Calculate daily ATR for volatility filter and position sizing
    # True Range for daily
    tr1_d = high - low
    tr2_d = np.abs(high - np.roll(close, 1))
    tr3_d = np.abs(low - np.roll(close, 1))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    tr_d[0] = tr1_d[0]
    
    atr_d = np.zeros_like(close)
    atr_d[14] = np.mean(tr_d[:15])  # 14-period ATR
    for i in range(15, len(close)):
        atr_d[i] = (atr_d[i-1] * 14 + tr_d[i]) / 15
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Volatility filter: avoid low volatility periods
    vol_ratio = atr_d / pd.Series(atr_d).rolling(window=50, min_periods=50).mean().values
    volatility_filter = vol_ratio > 0.8
    
    signals = np.zeros(n)
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume and volatility filters pass
        if volume_filter[i] and volatility_filter[i]:
            # Long conditions: weekly uptrend and price above Supertrend
            if direction_aligned[i] == 1 and close[i] > supertrend_aligned[i]:
                signals[i] = 0.25
            # Short conditions: weekly downtrend and price below Supertrend
            elif direction_aligned[i] == -1 and close[i] < supertrend_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklySupertrend_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0