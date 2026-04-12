#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d Regime Filter
    # Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to EMA
    # Bull Power = High - EMA, Bear Power = Low - EMA
    # In bull regimes (price > 1d EMA50): long when Bull Power > 0 and rising
    # In bear regimes (price < 1d EMA50): short when Bear Power < 0 and falling
    # Regime filter prevents counter-trend trading in strong trends
    # Works in bull/bear by adapting to the dominant trend via 1d EMA50
    
    # Session filter: 8:00-20:00 UTC (capture London/NY overlap)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA22 for Elder Ray (fast enough to react, slow enough to filter noise)
    close_s = pd.Series(close)
    ema_22 = close_s.ewm(span=22, adjust=False, min_periods=22).mean().values
    
    # Elder Ray components
    bull_power = high - ema_22   # Buying power: ability to push price above EMA
    bear_power = low - ema_22    # Selling power: ability to push price below EMA
    
    # Smooth the power signals to reduce noise (2-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_22[i]) or 
            np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination from 1d EMA50
        bull_regime = close[i] > ema_50_1d_aligned[i]
        bear_regime = close[i] < ema_50_1d_aligned[i]
        
        # Entry logic based on regime and Elder Ray
        if bull_regime:
            # In bull regime: long when bull power is positive and increasing
            long_entry = (bull_power_smooth[i] > 0) and (bull_power_smooth[i] > bull_power_smooth[i-1])
            # Exit when bull power turns negative (momentum fading)
            long_exit = bull_power_smooth[i] <= 0
        else:  # bear regime
            # In bear regime: short when bear power is negative and decreasing
            short_entry = (bear_power_smooth[i] < 0) and (bear_power_smooth[i] < bear_power_smooth[i-1])
            # Exit when bear power turns positive (selling pressure fading)
            short_exit = bear_power_smooth[i] >= 0
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0