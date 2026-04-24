#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend Filter + Volume Spike
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend direction.
- Williams %R (14) identifies overbought/oversold extremes: < -80 = oversold, > -20 = overbought.
- In bullish trend (close > EMA34): look for long entries when %R crosses above -80 from below (oversold bounce).
- In bearish trend (close < EMA34): look for short entries when %R crosses below -20 from above (overbought rejection).
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Exit: Opposite %R cross (%R < -80 for longs, %R > -20 for shorts) or EMA trend reversal.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets by aligning with 1d trend while picking extremes for entries.
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
    
    # Get 1d data for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA (34-period) on 1d close
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Williams %R (14-period) on 6h
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, lookback, 20)  # Need enough 1d bars for EMA and lookback for %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema_34_aligned[i]
        curr_close = close[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish trend: close > EMA34 -> look for oversold bounce
                if curr_close > ema_trend:
                    # Long when Williams %R crosses above -80 from below
                    if prev_williams <= -80 and curr_williams > -80:
                        signals[i] = 0.25
                        position = 1
                # Bearish trend: close < EMA34 -> look for overbought rejection
                elif curr_close < ema_trend:
                    # Short when Williams %R crosses below -20 from above
                    if prev_williams >= -20 and curr_williams < -20:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Williams %R drops below -80 (re-entered oversold) OR trend turns bearish
            if curr_williams < -80 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R rises above -20 (re-entered overbought) OR trend turns bullish
            if curr_williams > -20 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0