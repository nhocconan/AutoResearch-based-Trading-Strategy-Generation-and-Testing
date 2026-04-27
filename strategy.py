#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R mean reversion with 4h EMA trend filter and volume spike
# Williams %R identifies overbought/oversold conditions in ranging markets.
# 4h EMA provides trend direction to avoid counter-trend trades.
# Volume spike confirms institutional interest at reversal points.
# Session filter (08-20 UTC) reduces noise from low-volume periods.
# Target: 20-30 trades/year to minimize fee decay while capturing high-probability reversals.
# Works in both bull and bear markets by following higher timeframe trend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate Williams %R on 1h data (14-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    lookback = 14
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50
    
    # 24-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Warmup period
    start_idx = max(lookback, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine trend from 4h EMA20
        uptrend = price > ema_20_4h_aligned[i]
        downtrend = price < ema_20_4h_aligned[i]
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0 and in_session:
            # Long entry: oversold in uptrend with volume
            if uptrend and oversold and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: overbought in downtrend with volume
            elif downtrend and overbought and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: overbought or trend change
            if overbought or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: oversold or trend change
            if oversold or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_WilliamsR_4hEMA20_Volume_Session"
timeframe = "1h"
leverage = 1.0