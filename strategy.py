#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses 4h primary timeframe targeting 19-50 trades/year (75-200 total over 4 years).
# Camarilla R1/S1 levels from 1d: long when price breaks above R1 with volume, short when breaks below S1.
# 1d EMA34 provides trend filter: long only when price > EMA34, short only when price < EMA34.
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull and bear markets via trend filter + breakout logic.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot point = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Resistance levels: R1 = PP + Range * 1.1/12, R2 = PP + Range * 1.1/6, R3 = PP + Range * 1.1/4, R4 = PP + Range * 1.1/2
    r1 = pp + (range_1d * 1.1 / 12.0)
    r2 = pp + (range_1d * 1.1 / 6.0)
    r3 = pp + (range_1d * 1.1 / 4.0)
    r4 = pp + (range_1d * 1.1 / 2.0)
    # Support levels: S1 = PP - Range * 1.1/12, S2 = PP - Range * 1.1/6, S3 = PP - Range * 1.1/4, S4 = PP - Range * 1.1/2
    s1 = pp - (range_1d * 1.1 / 12.0)
    s2 = pp - (range_1d * 1.1 / 6.0)
    s3 = pp - (range_1d * 1.1 / 4.0)
    s4 = pp - (range_1d * 1.1 / 2.0)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction (price above/below EMA34)
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > r1_aligned[i] and volume_spike[i]
        short_breakout = close[i] < s1_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level (S1/R1) or trend reversal
        long_exit = close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]
        short_exit = close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if long_breakout and price_above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and price_below_ema and position >= 0:
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