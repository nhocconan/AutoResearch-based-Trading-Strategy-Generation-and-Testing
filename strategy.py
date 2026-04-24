#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA trend filter + volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend direction.
- Williams %R(14) identifies overbought/oversold conditions on 6h.
- In bullish 1d EMA(50) trend: look for long entries when Williams %R crosses above -80 (oversold bounce).
- In bearish 1d EMA(50) trend: look for short entries when Williams %R crosses below -20 (overbought rejection).
- Volume confirmation: current 6h volume > 1.3 * 20-period volume MA to avoid low-volume false signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: trend filter ensures we only take trades in direction of higher timeframe momentum.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Trend: 1 = bullish (close > EMA50), -1 = bearish (close < EMA50)
    trend = np.where(df_1d['close'].values > ema_50, 1, -1)
    
    # Align 1d trend to 6h
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend)
    
    # Calculate Williams %R(14) on 6h
    # Highest high over past 14 periods (including current)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    # Lowest low over past 14 periods (including current)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Range: -100 to 0, where -100 = oversold, 0 = overbought
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # Need enough for Williams %R, volume MA, and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trend_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_trend = trend_aligned[i]
        curr_wr = williams_r[i]
        curr_volume_ok = volume_spike[i]
        curr_close = close[i]
        prev_close = close[i-1]
        prev_wr = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals
            if curr_volume_ok:
                if curr_trend == 1:  # Bullish 1d trend: look for longs on Williams %R oversold bounce
                    # Long when Williams %R crosses above -80 (exiting oversold)
                    if prev_wr <= -80 and curr_wr > -80:
                        signals[i] = 0.25
                        position = 1
                elif curr_trend == -1:  # Bearish 1d trend: look for shorts on Williams %R overbought rejection
                    # Short when Williams %R crosses below -20 (exiting overbought)
                    if prev_wr >= -20 and curr_wr < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum weakening) OR trend turns bearish
            if curr_wr < -50 or curr_trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum weakening) OR trend turns bullish
            if curr_wr > -50 or curr_trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0