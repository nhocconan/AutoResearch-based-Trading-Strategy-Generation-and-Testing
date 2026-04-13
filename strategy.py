#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with weekly trend filter and daily volume confirmation.
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) identifies institutional buying/selling pressure.
# Weekly trend filter ensures we only take trades in the direction of the higher timeframe trend.
# Daily volume confirmation ensures institutional participation.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within profitable range.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily data for volume confirmation and EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA13 for trend filter
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate daily EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate daily high and low for Elder Ray
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 6-hour timeframe
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend: price above/below weekly EMA13
        weekly_uptrend = close[i] > ema13_1w_aligned[i]
        weekly_downtrend = close[i] < ema13_1w_aligned[i]
        
        # Daily Elder Ray signals
        strong_bull_power = bull_power_aligned[i] > 0
        strong_bear_power = bear_power_aligned[i] < 0
        
        # Volume condition: current 6h volume > 1.5x daily volume MA (adjusted for 6h)
        # 4 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 1.5)
        
        if position == 0:
            # Long: weekly uptrend + strong bull power + volume confirmation
            if weekly_uptrend and strong_bull_power and volume_condition:
                position = 1
                signals[i] = position_size
            # Short: weekly downtrend + strong bear power + volume confirmation
            elif weekly_downtrend and strong_bear_power and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when weekly trend turns down or bull power weakens
            if not weekly_uptrend or bull_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when weekly trend turns up or bear power weakens
            if not weekly_downtrend or bear_power_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Elder_Ray_Weekly_Trend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0