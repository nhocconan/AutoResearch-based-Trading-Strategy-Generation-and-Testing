#!/usr/bin/env python3
"""
12h Williams Alligator + Weekly Trend Filter
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trendless markets; 
when Alligator lines are intertwined (chop), avoid trades. When lines diverge 
(trending), take breakouts in direction of weekly EMA50 trend. 
Works in bull (long on upside breaks) and bear (short on downside breaks) 
by using weekly trend filter. Targets 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d := df_1d['close'].values, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set shifted values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align to 12h (no extra delay needed - Alligator known at 1d close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for volatility filter (14-period)
    def calculate_atr(high, low, close, period):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = np.full_like(tr, np.nan)
        for i in range(period, len(tr)):
            if i == period:
                atr[i] = np.nanmean(tr[1:period+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator and ATR
    start_idx = max(13, 14)  # Alligator needs 13, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        # Alligator condition: lines intertwined = chop (avoid trade)
        # Lines are intertwined when max-min < ATR * 0.5
        alligator_range = max(jaw_val, teeth_val, lips_val) - min(jaw_val, teeth_val, lips_val)
        chop_filter = alligator_range < (atr_val * 0.5)
        
        # Trend condition: price relative to weekly EMA
        above_weekly_ema = curr_close > ema_trend
        below_weekly_ema = curr_close < ema_trend
        
        # Breakout conditions: price breaks Alligator extremes
        alligator_high = max(jaw_val, teeth_val, lips_val)
        alligator_low = min(jaw_val, teeth_val, lips_val)
        
        breakout_up = curr_high > alligator_high
        breakout_down = curr_low < alligator_low
        
        if position == 0:
            # Look for entry signals only when not chopping
            if not chop_filter:
                # Long: breakout up + above weekly EMA
                if breakout_up and above_weekly_ema:
                    signals[i] = 0.25
                    position = 1
                # Short: breakout down + below weekly EMA
                elif breakout_down and below_weekly_ema:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Chop market, stay flat
        elif position == 1:
            # Long position management
            # Exit: breakout down OR price falls below weekly EMA
            if breakout_down or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: breakout up OR price rises above weekly EMA
            if breakout_up or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_WeeklyEMA_Trend_Filter"
timeframe = "12h"
leverage = 1.0