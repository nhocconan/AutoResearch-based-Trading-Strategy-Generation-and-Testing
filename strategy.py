# [131705] 12h_1dCamarilla_R1S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: 12h strategy using daily Camarilla pivot levels (R1/S1) for breakout entries,
# with 1d trend confirmation (EMA34) and volume spike confirmation. Exits on opposite touch
# (S1 for longs, R1 for shorts) or trend reversal. Designed to capture intraday momentum
# after pivot level breaks with institutional volume, working in both trending and ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing.

name = "12h_1dCamarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using prior day's range to avoid look-ahead
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # 1d EMA34 for trend direction
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_dir = np.where(close_1d > ema_34, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align 1d indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema34_dir_12h = align_htf_to_ltf(prices, df_1d, ema34_dir)
    
    # Volume spike filter (2x 20-period EMA on 12h)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema34_dir_12h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and bullish 1d trend
            if close[i] > r1_12h[i] and ema34_dir_12h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and bearish 1d trend
            elif close[i] < s1_12h[i] and ema34_dir_12h[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches S1 or trend turns bearish
            if close[i] < s1_12h[i] or ema34_dir_12h[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches R1 or trend turns bullish
            if close[i] > r1_12h[i] or ema34_dir_12h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1dCamarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # Using prior day's range to avoid look-ahead
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    
    # 1d EMA34 for trend direction
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_dir = np.where(close_1d > ema_34, 1, -1)  # 1=uptrend, -1=downtrend
    
    # Align 1d indicators to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    ema34_dir_12h = align_htf_to_ltf(prices, df_1d, ema34_dir)
    
    # Volume spike filter (2x 20-period EMA on 12h)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(ema34_dir_12h[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and bullish 1d trend
            if close[i] > r1_12h[i] and ema34_dir_12h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and bearish 1d trend
            elif close[i] < s1_12h[i] and ema34_dir_12h[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches S1 or trend turns bearish
            if close[i] < s1_12h[i] or ema34_dir_12h[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches R1 or trend turns bullish
            if close[i] > r1_12h[i] or ema34_dir_12h[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals