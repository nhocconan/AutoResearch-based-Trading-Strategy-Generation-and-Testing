#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation.
    # Camarilla levels provide precise support/resistance in ranging markets.
    # 4h EMA(20) filter ensures we trade with the higher timeframe trend.
    # Volume spike confirms breakout validity.
    # Session filter (08-20 UTC) reduces noise trades.
    # Discrete position sizing (0.0, ±0.20) minimizes fee churn.
    # Target: 60-150 total trades over 4 years (15-37/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot calculation and EMA trend filter (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot point (PP)
    pp = (high_4h + low_4h + close_4h) / 3.0
    # Range
    rng = high_4h - low_4h
    
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
    
    # Align Camarilla levels to 1h timeframe
    r4_1h = align_htf_to_ltf(prices, df_4h, r4)
    r3_1h = align_htf_to_ltf(prices, df_4h, r3)
    r2_1h = align_htf_to_ltf(prices, df_4h, r2)
    r1_1h = align_htf_to_ltf(prices, df_4h, r1)
    s1_1h = align_htf_to_ltf(prices, df_4h, s1)
    s2_1h = align_htf_to_ltf(prices, df_4h, s2)
    s3_1h = align_htf_to_ltf(prices, df_4h, s3)
    s4_1h = align_htf_to_ltf(prices, df_4h, s4)
    
    # Get 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(r4_1h[i]) or np.isnan(s4_1h[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(volume_ma[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: price above/below 4h EMA(20)
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Breakout conditions: price breaks Camarilla levels with volume and trend confirmation
        long_breakout = (close[i] > r4_1h[i-1]) and volume_filter and uptrend
        short_breakout = (close[i] < s4_1h[i-1]) and volume_filter and downtrend
        
        # Exit conditions: price returns to opposite Camarilla level
        long_exit = close[i] < s1_1h[i-1]
        short_exit = close[i] > r1_1h[i-1]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.20
        
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

name = "1h_4h_camarilla_breakout_trend_volume_session_v1"
timeframe = "1h"
leverage = 1.0