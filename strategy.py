#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray system with 1d trend filter and volume confirmation.
Alligator identifies market phase (sleeping/awake/hunting), Elder Ray measures bull/bear power,
1d EMA50 provides trend filter, volume > 2x average confirms strength.
Designed for low trade frequency (<40 total/year) to minimize fee drag in both bull and bear markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # SMA seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1d EMA50 to 4h timeframe (waits for 1d bar close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams Alligator: SMAs of median price
    # Jaw: 13-period SMMA, Teeth: 8-period, Lips: 5-period
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    def ema(arr, period):
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        multiplier = 2 / (period + 1)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = (arr[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 20-period average volume for spike detection
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need 50 for EMA50, 13 for EMA13, 20 for volume MA
    start_idx = max(50, 13, vol_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(jaw_aligned[i]) or
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: positive bull power = strength, negative bear power = weakness
        strong_bull = bull_power[i] > 0
        strong_bear = bear_power[i] < 0
        
        # Trend filter: price vs 1d EMA50
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        # Volume confirmation: > 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long entry: Alligator bullish + strong bull power + bullish trend + volume
            if bullish_alligator and strong_bull and bullish_trend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: Alligator bearish + strong bear power + bearish trend + volume
            elif bearish_alligator and strong_bear and bearish_trend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator turns bearish OR bear power becomes negative
            if not bullish_alligator or not strong_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Alligator turns bullish OR bull power becomes positive
            if not bearish_alligator or not strong_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Alligator_ElderRay_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0