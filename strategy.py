#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 12h Trend + Volume Confirmation
# Hypothesis: Elder Ray (bull/bear power) identifies institutional buying/selling pressure.
# Combined with 12h trend filter to trade in direction of higher timeframe momentum.
# Volume confirmation ensures institutional participation. Works in bull (buy power) and bear (sell power).
# Target: 15-35 trades/year to minimize fee drag.
name = "6h_elderay_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 12-period EMA for Elder Ray (using 6h data)
    ema_close = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_close  # High - EMA
    bear_power = low - ema_close   # Low - EMA
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily 20-period volume moving average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_close[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > daily average volume
        vol_confirm = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: 12h EMA direction
        uptrend = ema_12h_aligned[i] > ema_12h_aligned[i-1] if i > 0 else False
        downtrend = ema_12h_aligned[i] < ema_12h_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: bear power crosses above zero (selling pressure weakening) OR trend turns down
            if bear_power[i] > 0 or (downtrend and i > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: bull power crosses below zero (buying pressure weakening) OR trend turns up
            if bull_power[i] < 0 or (uptrend and i > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: bull power > 0 (buying pressure) + uptrend + volume confirmation
            if bull_power[i] > 0 and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: bear power < 0 (selling pressure) + downtrend + volume confirmation
            elif bear_power[i] < 0 and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals