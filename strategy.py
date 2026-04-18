#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold).
# Weekly trend filter ensures we only trade in direction of higher timeframe trend.
# Volume confirmation adds conviction to reversals.
# Designed for low trade frequency (15-30/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Williams %R (14-period) - momentum oscillator
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = (high_14 - close) / (high_14 - low_14) * -100
    williams_r = williams_r.values
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Weekly trend: price above/below EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND uptrend AND volume confirmation
            long_signal = williams_r[i] < -80
            if vol_confirm and uptrend and long_signal:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND downtrend AND volume confirmation
            elif vol_confirm and downtrend and williams_r[i] > -20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) OR trend changes
            exit_condition = williams_r[i] > -50 or not uptrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) OR trend changes
            exit_condition = williams_r[i] < -50 or not downtrend
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals