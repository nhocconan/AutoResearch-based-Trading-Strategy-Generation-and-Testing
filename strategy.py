#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter_v1
Hypothesis: Elder Ray Bull/Bear Power with 1d EMA trend filter and ATR-based stops.
Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
Long when Bull Power > 0 and Bear Power rising (less negative) + price > 1d EMA34.
Short when Bear Power > 0 and Bull Power falling (less positive) + price < 1d EMA34.
Uses 6h timeframe for execution, 1d for trend filter. Targets 12-35 trades/year.
Works in bull (trend following via EMA) and bear (mean reversion via Elder Ray extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray: EMA13 of close (6h timeframe)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    
    # Smooth Bull/Bear Power with EMA5 for rising/falling detection
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Rising Bull Power: current > previous
    bull_power_rising = bull_power_smooth > np.roll(bull_power_smooth, 1)
    bull_power_rising[0] = False
    # Falling Bear Power: current < previous (becoming less negative)
    bear_power_falling = bear_power_smooth < np.roll(bear_power_smooth, 1)
    bear_power_falling[0] = False
    
    # Rising Bear Power: current > previous
    bear_power_rising = bear_power_smooth > np.roll(bear_power_smooth, 1)
    bear_power_rising[0] = False
    # Falling Bull Power: current < previous (becoming less positive)
    bull_power_falling = bull_power_smooth < np.roll(bull_power_smooth, 1)
    bull_power_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after EMA13 and smoothing
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter from 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND rising + Bear Power falling (less negative) + uptrend
            long_signal = (bull_power[i] > 0) and bull_power_rising[i] and bear_power_falling[i] and uptrend
            # Short: Bear Power > 0 AND rising + Bull Power falling (less positive) + downtrend
            short_signal = (bear_power[i] > 0) and bear_power_rising[i] and bull_power_falling[i] and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Bear Power > 0 (momentum failure) OR stoploss
            # Calculate ATR for dynamic stop
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if bear_power[i] > 0:  # Bear Power positive = selling pressure
                signals[i] = 0.0
                position = 0
            elif curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power > 0 (momentum failure) OR stoploss
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if bull_power[i] > 0:  # Bull Power positive = buying pressure
                signals[i] = 0.0
                position = 0
            elif curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0