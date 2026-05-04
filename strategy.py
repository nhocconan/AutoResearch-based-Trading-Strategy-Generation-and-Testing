#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and volume confirmation
# Long when 1h EMA8 crosses above EMA21 AND 4h bullish trend (close > EMA34) AND volume > 1.5x 20-period volume EMA
# Short when 1h EMA8 crosses below EMA21 AND 4h bearish trend (close < EMA34) AND volume > 1.5x 20-period volume EMA
# Uses 4h EMA34 for trend filter to reduce whipsaw and align with higher timeframe momentum.
# Volume confirmation and session filter (08-20 UTC) reduce noise trades.
# Target: 15-37 trades/year on 1h by requiring confluence of trend, momentum, and volume.

name = "1h_EMA8_21_4hTrend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_4h = close_4h > ema_34_4h
    trend_bearish_4h = close_4h < ema_34_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Calculate 1h EMA8 and EMA21 for momentum
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade during active session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: EMA8 crosses above EMA21 AND 4h bullish trend AND volume spike
            if (ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1] and 
                trend_bullish_aligned[i] > 0.5 and  # 4h bullish trend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: EMA8 crosses below EMA21 AND 4h bearish trend AND volume spike
            elif (ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1] and 
                  trend_bearish_aligned[i] > 0.5 and  # 4h bearish trend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: EMA8 crosses below EMA21 OR 4h trend turns bearish
            if (ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]) or \
               trend_bearish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA8 crosses above EMA21 OR 4h trend turns bullish
            if (ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]) or \
               trend_bullish_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals