#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h for execution.
- Williams %R(14) identifies overbought/oversold conditions.
- Trend filter: 1d EMA(34) - price above EMA = bullish trend (favor longs), below = bearish (favor shorts).
- Volume confirmation: current 6h volume > 1.5 * 20-period volume MA to avoid false signals.
- In bullish trend (price > 1d EMA): Long when Williams %R crosses above -80 from below (exit oversold).
- In bearish trend (price < 1d EMA): Short when Williams %R crosses below -20 from above (exit overbought).
- Exit: Opposite Williams %R cross (%R < -80 for longs, %R > -20 for shorts) or trend reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets: Williams %R mean reversion + trend filter avoids counter-trend trades.
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams %R(14) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, lookback, 20)  # Need enough 1d bars for EMA and lookback for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_aligned[i]
        curr_close = close[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if curr_close > ema_val:  # Bullish trend: favor longs
                    # Long when Williams %R crosses above -80 from below (exit oversold)
                    if prev_williams <= -80 and curr_williams > -80:
                        signals[i] = 0.25
                        position = 1
                else:  # Bearish trend: favor shorts
                    # Short when Williams %R crosses below -20 from above (exit overbought)
                    if prev_williams >= -20 and curr_williams < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -80 (re-enter oversold) or trend turns bearish
            if curr_williams < -80 or curr_close < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -20 (re-enter overbought) or trend turns bullish
            if curr_williams > -20 or curr_close > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0