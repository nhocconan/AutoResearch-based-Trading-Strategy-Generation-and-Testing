#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (JAWS/TEETH/LIPS) identifies trending vs ranging markets.
In strong trends (JAWS > TEETH > LIPS for long, reverse for short), price tends to
continue in trend direction. Combined with 1d EMA50 for higher timeframe trend filter
and volume spike confirmation, this should capture sustained moves in both bull and
bear markets. Discrete sizing (0.25) targets ~75-150 trades over 4 years to minimize
fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h data (13,8,5 periods smoothed)
    # JAWS: 13-period SMMA shifted by 8 bars
    # TEETH: 8-period SMMA shifted by 5 bars  
    # LIPS: 5-period SMMA shifted by 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Shift the lines (JAWS: 8, TEETH: 5, LIPS: 3)
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # First 8 values of JAWS, first 5 of TEETH, first 3 of LIPS are invalid due to shift
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate ATR for stop loss (using 14 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Alligator (13+8=21) + EMA50 (50d) + ATR (14)
    start_idx = max(21, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaws_val = jaws[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        ema_trend = ema_50_aligned[i]
        atr_value = atr[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator conditions: JAWS > TEETH > LIPS (long) or JAWS < TEETH < LIPS (short)
        bullish_alligator = (jaws_val > teeth_val) and (teeth_val > lips_val)
        bearish_alligator = (jaws_val < teeth_val) and (teeth_val < lips_val)
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or Alligator reversal
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Alligator reversal or trend rejection
                elif not bullish_alligator or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Alligator reversal or trend rejection
                elif not bearish_alligator or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Alligator alignment + trend + volume
        if position == 0:
            # Long: bullish Alligator AND price above 1d EMA50
            long_condition = bullish_alligator and (curr_close > ema_trend) and volume_spike
            # Short: bearish Alligator AND price below 1d EMA50
            short_condition = bearish_alligator and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0