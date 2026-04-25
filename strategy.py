#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike + ATR Stoploss
Hypothesis: Williams Alligator (JAW/TEETH/LIPS) identifies trend absence/presence.
When Alligator is "sleeping" (JAW>TEETH>LIPS for down, LIPS>TEETH>JAW for up) and aligned
with 1d EMA50 trend, breakouts have higher follow-through. Volume spike confirms
institutional participation. ATR stoploss limits drawdown. Works in bull via trend
continuation and bear via avoiding counter-trend signals. Discrete sizing (0.25) minimizes fees.
Target: 12-30 trades/year on 12h timeframe.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = np.abs(np.diff(close, prepend=close[0]))
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = np.abs(high[0] - close[0])
        tr3[0] = np.abs(low[0] - close[0])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros(n)
        atr[:13] = np.nan
        for i in range(13, n):
            atr[i] = np.mean(tr[i-13:i+1])
    else:
        atr = np.full(n, np.nan)
    
    # Calculate 20-period volume MA for volume spike detection
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate Williams Alligator on 12h timeframe (SMMA with periods 13,8,5)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # SMMA (Smoothed Moving Average) calculation
    def smma(data, period):
        if len(data) < period:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: JAW (13,8), TEETH (8,5), LIPS (5,3)
    jaw_raw = smma(median_price_12h, 13)
    teeth_raw = smma(median_price_12h, 8)
    lips_raw = smma(median_price_12h, 5)
    
    # Align to lower timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA50_1d, ATR, Alligator, and volume MA to propagate
    start_idx = max(50, 13, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema50_1d = ema_50_1d_aligned[i]
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        # Alligator conditions:
        # Sleeping downtrend: JAW > TEETH > LIPS (all descending)
        # Sleeping uptrend: LIPS > TEETH > JAW (all ascending)
        sleeping_down = (jaw > teeth) and (teeth > lips)
        sleeping_up = (lips > teeth) and (teeth > jaw)
        
        if position == 0:
            # Long: Alligator sleeping uptrend AND price > JAW (breakout above) AND uptrend (close > 1d EMA50) AND volume spike
            long_condition = sleeping_up and (curr_close > jaw) and (curr_close > ema50_1d) and volume_spike
            # Short: Alligator sleeping downtrend AND price < JAW (breakdown below) AND downtrend (close < 1d EMA50) AND volume spike
            short_condition = sleeping_down and (curr_close < jaw) and (curr_close < ema50_1d) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or price breaks below TEETH (trend weakness)
            if curr_close <= entry_price - 2.5 * atr_val or curr_close < teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or price breaks above TEETH (trend weakness)
            if curr_close >= entry_price + 2.5 * atr_val or curr_close > teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0