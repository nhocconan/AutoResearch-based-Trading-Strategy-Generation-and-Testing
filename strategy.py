#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray (Bull/Bear Power) with 1w EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low.
# Enter long when Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) with volume spike and price > 1w EMA34.
# Enter short when Bear Power > 0 AND Bull Power < 0 (strong bearish momentum) with volume spike and price < 1w EMA34.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 12-37 trades/year (50-150 total over 4 years).
# Elder Ray measures momentum strength relative to EMA13, weekly EMA34 filters for higher timeframe trend,
# volume confirmation ensures breakout validity. Works in bull (strong momentum continuations) and bear (strong momentum reversals) markets.

name = "6h_ElderRay_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for Elder Ray calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Align 1w EMA34 to 6h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions with volume confirmation and weekly trend filter
        # Strong bullish: Bull Power > 0 AND Bear Power < 0 (both confirm upward momentum)
        strong_bullish = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        # Strong bearish: Bear Power > 0 AND Bull Power < 0 (both confirm downward momentum)
        strong_bearish = bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0
        
        # Entry conditions
        long_entry = strong_bullish and volume_spike[i] and close[i] > ema34_1w_aligned[i]
        short_entry = strong_bearish and volume_spike[i] and close[i] < ema34_1w_aligned[i]
        
        # Exit conditions: momentum weakening (opposite Elder Ray condition)
        long_exit = bear_power_aligned[i] > 0  # Bear Power turns positive
        short_exit = bull_power_aligned[i] > 0  # Bull Power turns positive
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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