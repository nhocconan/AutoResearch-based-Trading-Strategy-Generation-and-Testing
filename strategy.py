#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA50 trend filter and volume confirmation.
# Williams %R < -80 = oversold (long setup), > -20 = overbought (short setup).
# Only take reversals when price crosses EMA50 on 6h timeframe.
# 1d EMA50 must agree with trade direction (long only if price > 1d EMA50, short only if price < 1d EMA50).
# Volume must be > 1.3x 20-period MA to confirm participation.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).

name = "6h_WilliamsR_EMA50_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume confirmation: current 6h volume > 1.3x 20-period volume MA
        volume_confirm = volume[i] > (volume_ma_20[i] * 1.3)
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Price relative to 6h EMA50 for crossover confirmation
        ema_50_6h = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
        price_above_ema6h = close[i] > ema_50_6h[i]
        price_below_ema6h = close[i] < ema_50_6h[i]
        
        # 1d EMA50 trend filter
        price_above_ema1d = close[i] > ema_50_1d_aligned[i]
        price_below_ema1d = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold AND price crosses above 6h EMA50 AND price > 1d EMA50 AND volume confirm AND session
            if oversold and price_above_ema6h and price_above_ema1d and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought AND price crosses below 6h EMA50 AND price < 1d EMA50 AND volume confirm AND session
            elif overbought and price_below_ema6h and price_below_ema1d and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum fading) OR price crosses below 6h EMA50
            if williams_r[i] > -50 or price_below_ema6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 (momentum fading) OR price crosses above 6h EMA50
            if williams_r[i] < -50 or price_above_ema6h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals