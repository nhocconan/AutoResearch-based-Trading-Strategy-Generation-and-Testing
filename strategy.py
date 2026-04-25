#!/usr/bin/env python3
"""
6h Williams %R Reversal with 1d EMA34 Trend Filter and Volume Spike
Hypothesis: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe.
In strong 1d trends (price > EMA34 for longs, price < EMA34 for shorts), extreme %R readings
lead to high-probability reversals. Volume spike confirms institutional participation.
Designed for low trade frequency (12-37/year) with clear rules that work in both bull and bear markets
by only trading with the 1d trend and fading exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=np.float64)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_williams_r(high, low, close, period=14):
    """Calculate Williams %R"""
    if len(high) < period:
        return np.full_like(close, np.nan, dtype=np.float64)
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams_r.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = calculate_ema(df_1d['close'].values, 34)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 6h data
    williams_r = calculate_williams_r(high, low, close, 14)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R, EMA, volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R < -80 (oversold) AND volume spike AND price > 1d EMA34 (uptrend)
            long_entry = (wr < -80) and vol_spike and (curr_close > ema_trend)
            # Short: Williams %R > -20 (overbought) AND volume spike AND price < 1d EMA34 (downtrend)
            short_entry = (wr > -20) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R > -50 (return from oversold) OR price crosses below EMA (trend change)
            if (wr > -50) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R < -50 (return from overbought) OR price crosses above EMA (trend change)
            if (wr < -50) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0