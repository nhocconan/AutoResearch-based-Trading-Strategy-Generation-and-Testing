#!/usr/bin/env python3
name = "4h_Williams_Alligator_ElderRay_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 4h data for Williams Alligator
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Williams Alligator lines (median price)
    median_price_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    jaw = pd.Series(median_price_4h).rolling(window=13, center=False, min_periods=13).mean().values
    teeth = pd.Series(median_price_4h).rolling(window=8, center=False, min_periods=8).mean().values
    lips = pd.Series(median_price_4h).rolling(window=5, center=False, min_periods=5).mean().values
    
    # Align to lower timeframe
    jaw_4h = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_4h, teeth)
    lips_4h = align_htf_to_ltf(prices, df_4h, lips)
    
    # Elder Ray: Bull/Bear Power (13-period EMA of high/low minus EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Trend filter: 50 EMA on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator: Lips > Teeth > Jaw = bullish alignment
        # Lips < Teeth < Jaw = bearish alignment
        alligator_bull = lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i]
        alligator_bear = lips_4h[i] < teeth_4h[i] and teeth_4h[i] < jaw_4h[i]
        
        if position == 0:
            # Long: Bullish Alligator + Bull Power > 0 + price above 1d EMA50
            if alligator_bull and bull_power[i] > 0 and close[i] > ema_50_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + Bear Power < 0 + price below 1d EMA50
            elif alligator_bear and bear_power[i] < 0 and close[i] < ema_50_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator turns bearish or Bull Power turns negative
            if not alligator_bull or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator turns bullish or Bear Power turns positive
            if not alligator_bear or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals