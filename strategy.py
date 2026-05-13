# 165139
#!/usr/bin/env python3
"""
6h_Elder_Ray_Power_Divergence_1dTrend_Filter
Hypothesis: Elder Ray (Bull/Bear power) measures bull/bear strength relative to EMA13.
Divergence between price and Elder Ray signals exhaustion. Combined with 1d trend filter
(EMA34) to trade only in direction of higher timeframe trend. Low frequency (~15-25/year)
by requiring both divergence and trend alignment. Works in bull/bear via trend filter.
"""

name = "6h_Elder_Ray_Power_Divergence_1dTrend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13      # Bull power: high minus EMA
    bear_power = low - ema13       # Bear power: low minus EMA
    
    # 6-period smoothed Elder Ray for divergence detection
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Divergence signals: price makes new high/low but power doesn't confirm
    # Bullish divergence: price makes lower low, bull power makes higher low
    # Bearish divergence: price makes higher high, bear power makes lower high
    price_lower_low = (low < np.roll(low, 1)) & (np.roll(low, 1) < np.roll(low, 2))
    bull_power_higher_low = (bull_power_smooth > np.roll(bull_power_smooth, 1)) & (np.roll(bull_power_smooth, 1) > np.roll(bull_power_smooth, 2))
    bull_divergence = price_lower_low & bull_power_higher_low
    
    price_higher_high = (high > np.roll(high, 1)) & (np.roll(high, 1) > np.roll(high, 2))
    bear_power_lower_high = (bear_power_smooth < np.roll(bear_power_smooth, 1)) & (np.roll(bear_power_smooth, 1) < np.roll(bear_power_smooth, 2))
    bear_divergence = price_higher_high & bear_power_lower_high
    
    # 1d trend filter: EMA34 on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: avoid low volatility periods
    vol_ma = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_filter = volume > (0.5 * vol_ma)  # at least half average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        if position == 0:
            # LONG: Bullish divergence + price above 1d EMA34 (uptrend) + volume filter
            if bull_divergence[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + price below 1d EMA34 (downtrend) + volume filter
            elif bear_divergence[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish divergence appears OR price closes below EMA13
            if bear_divergence[i] or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish divergence appears OR price closes above EMA13
            if bull_divergence[i] or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals