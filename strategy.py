#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Keltner_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_6h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Keltner Channel (20, 2.0) on daily timeframe
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    atr = pd.Series(
        np.maximum(
            np.maximum(df_1d['high'] - df_1d['low'], 
                      np.abs(df_1d['high'] - df_1d['close'].shift(1))),
            np.abs(df_1d['low'] - df_1d['close'].shift(1))
        )
    ).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    ema_tp = typical_price.ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_tp + 2.0 * atr
    lower_keltner = ema_tp - 2.0 * atr
    
    # Align Keltner levels to 6h
    upper_keltner_6h = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_6h = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_tp_6h = align_htf_to_ltf(prices, df_1d, ema_tp)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_6h[i]) or np.isnan(lower_keltner_6h[i]) or 
            np.isnan(ema20_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above upper Keltner with weekly uptrend and volume spike
            if close[i] > upper_keltner_6h[i] and close[i] > ema20_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Keltner with weekly downtrend and volume spike
            elif close[i] < lower_keltner_6h[i] and close[i] < ema20_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Keltner middle OR weekly trend turns down
            if close[i] < ema_tp_6h[i] or close[i] < ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Keltner middle OR weekly trend turns up
            if close[i] > ema_tp_6h[i] or close[i] > ema20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals