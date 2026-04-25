#!/usr/bin/env python3
"""
1d Williams Alligator Breakout with 1w EMA50 Trend and Volume Spike v1
Hypothesis: Williams Alligator (jaw/teeth/lips) on 1d acts as dynamic support/resistance.
Breakouts above lips or below jaw with volume confirmation (>1.8x 20-bar vol MA) and 1w EMA50 trend
filter capture strong momentum moves. Uses ATR-based trailing stop (2.0*ATR) for risk control.
Tight entry conditions target 30-100 total trades over 4 years to avoid fee drag. Works in
bull markets via long breakouts and in bear markets via short breakdowns. EMA50 on 1w provides
smoother trend filter than shorter EMAs, reducing whipsaws in choppy markets and improving
generalization to bear markets (2025+ test period).
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
    
    # Calculate Williams Alligator on 1d: jaw (13,8), teeth (8,5), lips (5,3)
    # Alligator lines are smoothed medians (SMMA) of typical price
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - same as RMA/Wilder's"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(typical_price.values, 13)
    teeth = smma(typical_price.values, 8)
    lips = smma(typical_price.values, 5)
    
    # Align Alligator lines to 1d timeframe (they are already 1d, but need alignment to 1d bars)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period volume MA for volume confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (1d)
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_stop = 0.0
    
    # Start index: need enough for Alligator, EMA50_1w, volume MA, ATR to propagate
    start_idx = max(13, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50_1w = ema_50_1w_aligned[i]
        vol_ma = vol_ma_20[i]
        atr = atr_14[i]
        
        # Volume confirmation: current volume > 1.8 * 20-period average (moderate filter)
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Long breakout: close above lips with volume confirmation and 1w EMA50 uptrend
            long_breakout = (curr_close > lips_val) and volume_confirm and (curr_close > ema50_1w)
            # Short breakdown: close below jaw with volume confirmation and 1w EMA50 downtrend
            short_breakout = (curr_close < jaw_val) and volume_confirm and (curr_close < ema50_1w)
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_stop = curr_close - 2.0 * atr  # Initial stop
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_stop = curr_close + 2.0 * atr  # Initial stop
        elif position == 1:
            # Update trailing stop: raise stop to highest high - 2.0*ATR
            atr_stop = max(atr_stop, curr_high - 2.0 * atr)
            # Exit long: price closes below trailing stop
            if curr_close < atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update trailing stop: lower stop to lowest low + 2.0*ATR
            atr_stop = min(atr_stop, curr_low + 2.0 * atr)
            # Exit short: price closes above trailing stop
            if curr_close > atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Williams_Alligator_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0