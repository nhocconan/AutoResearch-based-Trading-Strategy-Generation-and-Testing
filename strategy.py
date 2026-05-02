#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for HTF trend alignment to reduce whipsaw vs shorter timeframes
# Williams Alligator (Jaw/Teeth/Lips) provides clear trend structure and entry signals
# Breakout above Lips with volume confirms bullish momentum, breakdown below Lips confirms bearish
# 1d EMA50 trend filter ensures alignment with daily trend to avoid counter-trend trades
# Works in both bull and bear markets by following daily trend with Alligator signals
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Discrete position sizing: 0.25 (25% of capital) to minimize fee churn while maintaining reasonable exposure

name = "12h_Williams_Alligator_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price = (H + L + C) / 3
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Alligator components: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    jaw_1d = pd.Series(typical_price_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth_1d = pd.Series(typical_price_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    lips_1d = pd.Series(typical_price_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price crosses above Lips with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > lips_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below Lips with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < lips_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Lips OR below 1d EMA50 (trend change)
            if close[i] < lips_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Lips OR above 1d EMA50 (trend change)
            if close[i] > lips_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals