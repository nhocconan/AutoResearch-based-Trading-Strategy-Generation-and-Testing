#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA50 trend direction.
- Williams %R(14) < -80 indicates oversold (long setup), > -20 indicates overbought (short setup).
- In uptrend (close > EMA50 on 1d): take longs on Williams %R < -80 with reversal confirmation.
- In downtrend (close < EMA50 on 1d): take shorts on Williams %R > -20 with reversal confirmation.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R extreme or trend reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R (14-period) on 4h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 1d bars for EMA50 and lookback for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        williams_val = williams_r[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema50_val:  # Uptrend: look for longs on oversold
                    # Long when Williams %R < -80 (oversold) and price reverses up (close > low)
                    if williams_val < -80 and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend: look for shorts on overbought
                    # Short when Williams %R > -20 (overbought) and price reverses down (close < high)
                    if williams_val > -20 and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR trend reversal to downtrend
            if williams_val > -20 or curr_close < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR trend reversal to uptrend
            if williams_val < -80 or curr_close > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0