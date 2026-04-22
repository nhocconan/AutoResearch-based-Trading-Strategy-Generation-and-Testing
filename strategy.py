#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price > 1w EMA50, and volume spike.
Short when Bull Power < 0, Bear Power > 0, price < 1w EMA50, and volume spike.
Exit when Bull Power and Bear Power converge (both near zero) or trend changes.
Designed for low trade frequency (15-30/year) to minimize fee drag in 6h timeframe.
Elder Ray captures bull/bear strength via EMA13; weekly trend filter avoids counter-trend trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate EMA13 for Elder Ray (13-period on 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after volume lookback
        # Skip if data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > 1w EMA50, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price < 1w EMA50, volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend change or power convergence
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power >= 0 (bulls weakening) or trend turns bearish
                if bear_power[i] >= 0 or close[i] < ema50_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power <= 0 (bears weakening) or trend turns bullish
                if bull_power[i] <= 0 or close[i] > ema50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1wEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0
#%%