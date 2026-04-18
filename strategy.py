#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_WeeklyTrend
Hypothesis: Elder Ray (Bull/Bear Power) combined with weekly trend filter (weekly EMA200) to capture momentum in both bull and bear markets.
- Bull Power = High - EMA13 (13-period EMA of close)
- Bear Power = Low - EMA13 (13-period EMA of close)
- Long when Bull Power > 0 and Bear Power rising (momentum) and weekly trend up
- Short when Bear Power < 0 and Bull Power falling and weekly trend down
- Uses weekly EMA200 for trend filter to avoid counter-trend trades
- Target: 15-30 trades/year to minimize fee drag while capturing sustained moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA200
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Momentum: rate of change of Bull/Bear Power (3-period)
    bull_power_series = pd.Series(bull_power)
    bear_power_series = pd.Series(bear_power)
    bull_power_momentum = bull_power_series.diff(3).values
    bear_power_momentum = bear_power_series.diff(3).values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(100, 13 + 3)  # Warmup for EMA13 and momentum
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(bull_power_momentum[i]) or
            np.isnan(bear_power_momentum[i])):
            signals[i] = 0.0
            continue
        
        bull_pwr = bull_power[i]
        bear_pwr = bear_power[i]
        bull_mom = bull_power_momentum[i]
        bear_mom = bear_power_momentum[i]
        weekly_ema = ema200_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power positive AND rising (momentum up) AND weekly uptrend
            if bull_pwr > 0 and bull_mom > 0 and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND falling (momentum down) AND weekly downtrend
            elif bear_pwr < 0 and bear_mom < 0 and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: Bull Power turns negative OR weekly trend breaks down
            if bull_pwr <= 0:
                signals[i] = 0.0
                position = 0
            elif price < weekly_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: Bear Power turns positive OR weekly trend breaks up
            if bear_pwr >= 0:
                signals[i] = 0.0
                position = 0
            elif price > weekly_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_WeeklyTrend"
timeframe = "6h"
leverage = 1.0