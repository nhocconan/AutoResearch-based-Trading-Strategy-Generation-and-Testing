#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA50 trend direction.
- Williams %R(14) < -80 = oversold, > -20 = overbought.
- In uptrend (close > EMA50): Long when %R crosses above -80 from below (oversold bounce).
- In downtrend (close < EMA50): Short when %R crosses below -20 from above (overbought rejection).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite %R crossover or EMA50 trend flip.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets by following the 1d trend while using 6h mean reversion entries.
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
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
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
    start_idx = max(50, lookback, 20)  # Need enough 1d bars for EMA50 and lookback for %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_50_aligned[i]
        curr_close = close[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema_trend:  # Uptrend: look for oversold bounce
                    # Long when Williams %R crosses above -80 from below
                    if prev_williams < -80 and curr_williams >= -80:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend: look for overbought rejection
                    # Short when Williams %R crosses below -20 from above
                    if prev_williams > -20 and curr_williams <= -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 OR EMA50 trend flips down
            if curr_williams < -50 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR EMA50 trend flips up
            if curr_williams > -50 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0