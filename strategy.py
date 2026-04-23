#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d regime filter.
- Primary timeframe: 6h, HTF: 1d for trend regime
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
- Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power > 0 + Bear Power < 0 (strong bull)
- Short: Alligator aligned inverse (Lips < Teeth < Jaw) + Bear Power < 0 + Bull Power < 0 (strong bear)
- Exit: Alligator alignment breaks or Elder Ray power weakens
- Uses discrete sizing ±0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
- Works in trending markets (Alligator alignment) and filters choppy conditions
"""

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
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator SMAs (using median price)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # Jaw (13)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # Teeth (8)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # Lips (5)
    
    # Elder Ray: Bull/Bear Power using EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d trend regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ADX for trend strength (optional filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    plus_di = 100 * wilders_smoothing(plus_dm, period) / wilders_smoothing(tr, period)
    minus_di = 100 * wilders_smoothing(minus_dm, period) / wilders_smoothing(tr, period)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 1)  # Need 13 for Alligator Jaw
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips[i] > teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray power conditions
        strong_bull = bull_power[i] > 0 and bear_power[i] < 0
        strong_bear = bear_power[i] < 0 and bull_power[i] < 0
        
        # Regime filter: trending market (ADX > 25) + 1d EMA direction
        trending = adx_aligned[i] > 25
        uptrend_1d = close[i] > ema_34_aligned[i]
        downtrend_1d = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator + strong bull power + uptrend regime
            if (bullish_alignment and strong_bull and trending and uptrend_1d):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator + strong bear power + downtrend regime
            elif (bearish_alignment and strong_bear and trending and downtrend_1d):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR bear power becomes positive
            if not bullish_alignment or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR bull power becomes positive
            if not bearish_alignment or bull_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_ADX"
timeframe = "6h"
leverage = 1.0