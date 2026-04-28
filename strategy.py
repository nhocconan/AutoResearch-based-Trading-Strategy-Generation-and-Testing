#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
# Enter long when Bull Power > 0 and Bear Power < 0 (strong bullish pressure) with volume > 1.5x 20-bar average and price > 1w EMA34 (uptrend).
# Enter short when Bear Power > 0 and Bull Power < 0 (strong bearish pressure) with volume > 1.5x 20-bar average and price < 1w EMA34 (downtrend).
# Uses discrete position sizing (0.25) to balance return and drawdown. Target: 12-37 trades/year.
# Elder Ray measures price power relative to EMA, volume confirms conviction, 1w EMA34 filters for higher timeframe trend alignment.
# Works in bull (trend continuation) and bear (trend continuation) markets by following the 1w trend.

name = "6h_ElderRay_1wEMA34_Trend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Calculate 1d Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d  # High - EMA13
    bear_power = ema13_1d - low_1d   # EMA13 - Low
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Align 1w EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions with volume confirmation and 1w trend filter
        long_condition = (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                         volume_confirm[i] and close[i] > ema34_1w_aligned[i])
        short_condition = (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                          volume_confirm[i] and close[i] < ema34_1w_aligned[i])
        
        # Exit conditions: opposite Elder Ray signal (loss of power)
        long_exit = bull_power_aligned[i] < 0  # Bull power turned negative
        short_exit = bear_power_aligned[i] < 0  # Bear power turned negative
        
        # Handle entries and exits
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals