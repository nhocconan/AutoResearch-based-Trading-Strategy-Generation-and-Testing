#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation.
    # Camarilla levels provide precise support/resistance in ranging markets.
    # 1w EMA filter ensures we trade with the higher timeframe trend.
    # Volume spike confirms breakout validity.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 30-100 total trades over 4 years (7-25/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range
    rng = high_1d - low_1d
    
    # Camarilla levels
    # Resistance levels
    r4 = pp + (rng * 1.1 / 2.0)
    r3 = pp + (rng * 1.1 / 4.0)
    r2 = pp + (rng * 1.1 / 6.0)
    r1 = pp + (rng * 1.1 / 12.0)
    # Support levels
    s1 = pp - (rng * 1.1 / 12.0)
    s2 = pp - (rng * 1.1 / 6.0)
    s3 = pp - (rng * 1.1 / 4.0)
    s4 = pp - (rng * 1.1 / 2.0)
    
    # Align Camarilla levels to 1d timeframe
    r4_1d = align_htf_to_ltf(prices, df_1d, r4)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    s4_1d = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1w EMA to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below 1w EMA(20)
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend confirmation
        long_breakout = (close[i] > r4_1d[i-1]) and volume_filter and uptrend
        short_breakout = (close[i] < s4_1d[i-1]) and volume_filter and downtrend
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < s1_1d[i-1]
        short_exit = close[i] > r1_1d[i-1]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_trend_volume_v1"
timeframe = "1d"
leverage = 1.0