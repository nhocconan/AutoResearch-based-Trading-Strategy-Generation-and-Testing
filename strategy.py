#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with Elder Ray force index and weekly trend filter.
# Long when: Green line > Red line (bullish alignment), Elder Ray Bull Power > 0, and price > weekly EMA50.
# Short when: Red line > Green line (bearish alignment), Elder Ray Bear Power < 0, and price < weekly EMA50.
# Uses Williams Alligator (JAWS=13, TEETH=8, LIPS=5) smoothed with SMMA.
# Elder Ray uses 13-period EMA for Bull/Bear Power calculation.
# Weekly EMA50 acts as trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year by requiring triple alignment (Alligator + Elder Ray + weekly trend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_w = df_1w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 4h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    
    # Calculate Williams Alligator components (SMMA)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    median_price = (high + low) / 2
    
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.nansum(arr[:period]) / period
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line
    
    # Calculate Elder Ray Force Index components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(ema50_1w_aligned[i]):
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
        
        # Williams Alligator signals
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaws[i]
        bearish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        # Weekly trend filter
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Enter long on bullish alignment + bull power + price above weekly EMA
            if bullish_alignment and bull_power_positive and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Enter short on bearish alignment + bear power + price below weekly EMA
            elif bearish_alignment and bear_power_negative and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reversal of alignment or loss of power
            exit_signal = False
            
            if position == 1:
                # Exit long: bearish alignment OR bull power turns negative
                if bearish_alignment or bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: bullish alignment OR bear power turns positive
                if bullish_alignment or bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "4h"
leverage = 1.0