#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with Elder Ray and 1d trend filter.
# Long when: Alligator bullish (JAW > TEETH > LIPS) + Bull Power > 0 + price > 1d EMA50.
# Short when: Alligator bearish (LIPS < TEETH < JAW) + Bear Power < 0 + price < 1d EMA50.
# Exit when Alligator direction changes or Elder Power reverses.
# Uses 1d EMA50 trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year by requiring Alligator alignment + Elder confirmation + trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d for EMA50 trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA13 for Elder Ray (standard period)
    close_d = df_1d['close'].values
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    ema13_d = pd.Series(close_d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_d = high_d - ema13_d
    bear_power_d = low_d - ema13_d
    
    # Align 1d indicators to 6h
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_d)
    
    # Calculate 6h Williams Alligator (SMMA: 13, 8, 5)
    close = prices['close'].values
    # SMMA (Smoothed Moving Average) approximation using EMA with alpha=1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        res = np.full_like(arr, np.nan)
        alpha = 1.0 / period
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (arr[i] * alpha) + (res[i-1] * (1 - alpha))
        return res
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        alligator_bullish = jaw[i] > teeth[i] > lips[i]
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray conditions
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        
        # Trend filter: price vs daily EMA13
        price = prices['close'].iloc[i]
        bull_trend = price > ema13_1d_aligned[i]
        bear_trend = price < ema13_1d_aligned[i]
        
        if position == 0:
            # Enter long on Alligator bullish + Bull Power positive + bullish trend
            if alligator_bullish and bull_power > 0 and bull_trend:
                signals[i] = 0.25
                position = 1
            # Enter short on Alligator bearish + Bear Power negative + bearish trend
            elif alligator_bearish and bear_power < 0 and bear_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator direction change or Elder Power reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR Bull Power turns negative
                if not alligator_bullish or bull_power <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Alligator turns bullish OR Bear Power turns positive
                if not alligator_bearish or bear_power >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dTrend"
timeframe = "6h"
leverage = 1.0