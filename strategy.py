#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w regime filter
# Elder Ray (bull/bear power) captures trend strength using EMA13 as reference
# Weekly trend filter (price vs EMA40) avoids counter-trend trades in strong trends
# Works in bull/bear by only taking longs in bull regime and shorts in bear regime
# Target: 20-60 trades/year with clear entry/exit conditions to minimize churn

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: price vs EMA40 on weekly timeframe
    weekly = get_htf_data(prices, '1w')
    weekly_close = weekly['close'].values
    weekly_ema40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema40_aligned = align_htf_to_ltf(prices, weekly, weekly_ema40)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(40, n):  # Start after warmup for weekly EMA40
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(weekly_ema40_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine weekly regime: bull if price > weekly EMA40, bear if price < weekly EMA40
        weekly_bull = close[i] > weekly_ema40_aligned[i]
        weekly_bear = close[i] < weekly_ema40_aligned[i]
        
        # Only trade with volume confirmation
        if volume_filter[i]:
            # Long conditions: bull regime + bull power positive and increasing
            if weekly_bull and bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                signals[i] = 0.25
            # Short conditions: bear regime + bear power negative and decreasing (more negative)
            elif weekly_bear and bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0  # No position when conditions not met
        else:
            signals[i] = 0.0  # No position when volume filter fails
    
    return signals

name = "6h_ElderRay_WeeklyEMA40_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0