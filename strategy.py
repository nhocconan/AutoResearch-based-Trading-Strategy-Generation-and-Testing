#!/usr/bin/env python3
"""
12h_Williams_Alligator_ElderRay_Filter
Hypothesis: Combines Williams Alligator (Jaw/Teeth/Lips) for trend direction with
Elder Ray's Bull/Bear Power for momentum confirmation on 12h timeframe. Uses weekly
trend filter to avoid counter-trend trades. Designed to work in both bull and bear
markets by only taking trades in the direction of the higher timeframe trend.
Target: 12-30 trades/year per symbol.
"""

name = "12h_Williams_Alligator_ElderRay_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Convert to Series for indicator calculations
    close_s = pd.Series(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    volume_s = pd.Series(volume)
    
    # Williams Alligator: SMMA (Smoothed MA) with periods 13, 8, 5
    # Jaw (blue): 13-period SMMA, 8 bars ahead
    # Teeth (red): 8-period SMMA, 5 bars ahead  
    # Lips (green): 5-period SMMA, 3 bars ahead
    def smma(values, period):
        """Smoothed Moving Average"""
        sma = pd.Series(values).rolling(window=period, min_periods=period).mean()
        # Initialize first value as SMA
        result = np.full_like(values, np.nan, dtype=float)
        if len(sma) >= period:
            result[period-1] = sma.iloc[period-1]
            for i in range(period, len(values)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + values[i]) / period
                else:
                    result[i] = np.nan
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift according to Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray: Bull Power and Bear Power using 13-period EMA
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align weekly trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: 20-period average
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Enter long: bullish alignment + bullish momentum + weekly uptrend + volume
            if (alligator_bullish and bull_power[i] > 0 and 
                trend_1w_up_aligned[i] > 0.5 and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment + bearish momentum + weekly downtrend + volume
            elif (alligator_bearish and bear_power[i] < 0 and 
                  trend_1w_down_aligned[i] > 0.5 and volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when alignment breaks or momentum fades
            if not alligator_bullish or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when alignment breaks or momentum fades
            if not alligator_bearish or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals