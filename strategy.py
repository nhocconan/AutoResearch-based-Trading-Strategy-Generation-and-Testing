#!/usr/bin/env python3
# 4h_Williams_Alligator_Elder_Ray_Momentum
# Hypothesis: Williams Alligator defines market structure (jaws/teeth/lips), Elder Ray measures bull/bear power.
# Long when price > Alligator teeth + Bull Power > 0 + Bear Power < 0 + volume confirmation.
# Short when price < Alligator teeth + Bull Power < 0 + Bear Power > 0 + volume confirmation.
# Uses 13/8/5 SMAs for Alligator, 13-period EMA for Elder Ray. Filters reduce false signals in ranging markets.
# Designed for moderate trade frequency (target: 20-50 trades/year) with trend strength confirmation.

name = "4h_Williams_Alligator_Elder_Ray_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Williams Alligator: SMAs of median price
    median_price = (high + low) / 2
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator (13), Elder Ray (13), EMA (34), volume (20)
    start_idx = max(13, 13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(teeth[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator condition: price vs teeth (8-period SMA)
        price_above_teeth = close[i] > teeth[i]
        price_below_teeth = close[i] < teeth[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 for long, vice versa for short
        bull_positive = bull_power[i] > 0
        bear_negative = bear_power[i] < 0
        bull_negative = bull_power[i] < 0
        bear_positive = bear_power[i] > 0
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long entry: price above teeth + bull power positive + bear power negative + volume + uptrend
            if price_above_teeth and bull_positive and bear_negative and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price below teeth + bull power negative + bear power positive + volume + downtrend
            elif price_below_teeth and bull_negative and bear_positive and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below teeth OR Elder Ray turns bearish OR trend breaks
            if close[i] < teeth[i] or not (bull_positive and bear_negative) or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above teeth OR Elder Ray turns bullish OR trend breaks
            if close[i] > teeth[i] or not (bull_negative and bear_positive) or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals