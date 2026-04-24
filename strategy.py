#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 12h EMA Trend Filter + Volume Spike
- Primary timeframe: 6h for execution, HTF: 12h for EMA trend direction.
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought.
- In uptrend (price > 12h EMA50): look for oversold Williams %R (< -80) with volume spike for long entries.
- In downtrend (price < 12h EMA50): look for overbought Williams %R (> -20) with volume spike for short entries.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R extreme or trend reversal (price crosses 12h EMA50).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets by adapting to trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 12h bars for EMA and lookback for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_50_aligned[i]
        curr_close = close[i]
        curr_low = low[i]
        curr_high = high[i]
        wr = williams_r[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema_trend:  # Uptrend: look for oversold bounce
                    if wr < -80:  # Oversold condition
                        signals[i] = 0.25
                        position = 1
                elif curr_close < ema_trend:  # Downtrend: look for overbought rejection
                    if wr > -20:  # Overbought condition
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R becomes overbought OR trend turns down
            if wr > -20 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R becomes oversold OR trend turns up
            if wr < -80 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0