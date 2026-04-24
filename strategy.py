#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme + 1w EMA trend filter + volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA trend direction.
- Williams %R(14) < -80 = oversold (long signal), > -20 = overbought (short signal).
- Only take longs when price > 1w EMA34 (bullish trend), shorts when price < 1w EMA34 (bearish trend).
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite Williams %R extreme (%R > -50 for longs, %R < -50 for shorts) or EMA trend flip.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull markets (trend + oversold bounces) and bear markets (trend + overbought rejections).
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
    
    # Get 1w data for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R (14-period) on 12h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback, 20)  # Need enough 1w bars for EMA and lookback for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_34_1w_aligned[i]
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Bullish: oversold AND above 1w EMA (bullish trend)
                if curr_williams_r < -80 and curr_close > ema_trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish: overbought AND below 1w EMA (bearish trend)
                elif curr_williams_r > -20 and curr_close < ema_trend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R exits oversold (> -50) OR price crosses below 1w EMA
            if curr_williams_r > -50 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R exits overbought (< -50) OR price crosses above 1w EMA
            if curr_williams_r < -50 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1wEMA34Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0