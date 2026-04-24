#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R reversal with 1d volume spike and 1w EMA trend filter.
- Primary timeframe: 12h for execution, HTF: 1d for Williams %R (oversold/overbought), 1w for EMA50 trend direction.
- Williams %R < -80 indicates oversold (long setup), > -20 indicates overbought (short setup).
- EMA50 > EMA200 on 1w indicates bullish trend (favor longs), EMA50 < EMA200 indicates bearish trend (favor shorts).
- Entry: Long when Williams %R crosses above -80 AND EMA50 > EMA200 (bullish trend + oversold bounce).
         Short when Williams %R crosses below -20 AND EMA50 < EMA200 (bearish trend + overbought rejection).
- Exit: Opposite Williams %R crossover (cross below -50 for long, cross above -50 for short) or EMA trend flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid low-volume false signals).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 1d
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10)
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on 1w
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_bullish = ema_50 > ema_200  # True when bullish trend
    
    # Align HTF indicators to 12h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA200 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_bullish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_wr = williams_r_aligned[i]
        prev_wr = williams_r_aligned[i-1]
        is_ema_bullish = ema_bullish_aligned[i] > 0.5
        curr_close = close[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if is_ema_bullish:  # Bullish trend: look for longs on oversold bounce
                    # Long when Williams %R crosses above -80 (exiting oversold)
                    if prev_wr <= -80 and curr_wr > -80:
                        signals[i] = 0.25
                        position = 1
                else:  # Bearish trend: look for shorts on overbought rejection
                    # Short when Williams %R crosses below -20 (exiting overbought)
                    if prev_wr >= -20 and curr_wr < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum fading) OR EMA trend flips bearish
            if curr_wr < -50 and prev_wr >= -50:
                signals[i] = 0.0
                position = 0
            elif not is_ema_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum fading) OR EMA trend flips bullish
            if curr_wr > -50 and prev_wr <= -50:
                signals[i] = 0.0
                position = 0
            elif is_ema_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dVolumeSpike_1wEMATrend_v1"
timeframe = "12h"
leverage = 1.0