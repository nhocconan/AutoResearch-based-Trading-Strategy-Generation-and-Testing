#!/usr/bin/env python3
"""
6h Williams Alligator + 1d EMA50 Trend + Volume Confirmation
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence. 
When Lips cross above Teeth (bullish) or below Teeth (bearish) with 1d EMA50 trend alignment 
and volume confirmation, it captures the start of sustained moves. Works in bull markets 
via buying bullish Alligator crosses above EMA50, bear markets via selling bearish crosses 
below EMA50. Discrete position sizing (0.25) controls drawdown. Target: 12-37 trades/year 
on 6h (50-150 total over 4 years).
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    if len(close) >= 14:
        tr1 = pd.Series(high).diff().abs()
        tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
        tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=14).mean().values
    else:
        atr = np.full(n, 0.0)
    
    # Williams Alligator: SMAs of median price (HLC/3)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    median_price = (high + low + close) / 3.0
    
    def smma(source, period):
        """Smoothed Moving Average (SMMA) aka Wilder's MA"""
        if len(source) < period:
            return np.full(len(source), np.nan)
        result = np.full(len(source), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (prev_SMMA*(period-1) + current_price) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Pre-compute 20-period volume MA for volume confirmation
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Alligator SMAs (13-period) and EMA50 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_lips = lips[i]
        curr_teeth = teeth[i]
        curr_jaw = jaw[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Alligator signals: Lips above Teeth = bullish, Lips below Teeth = bearish
        bullish_alligator = curr_lips > curr_teeth
        bearish_alligator = curr_lips < curr_teeth
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: bullish Alligator AND price above EMA50 AND volume confirmation
            long_condition = bullish_alligator and curr_close > ema_50 and volume_confirm
            # Short: bearish Alligator AND price below EMA50 AND volume confirmation
            short_condition = bearish_alligator and curr_close < ema_50 and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.5*ATR below entry) or Alligator turns bearish
            if curr_close <= entry_price - 2.5 * atr_val or curr_lips < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.5*ATR above entry) or Alligator turns bullish
            if curr_close >= entry_price + 2.5 * atr_val or curr_lips > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0