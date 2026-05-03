#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Volume Spike + 1w Trend Filter
# Elder Ray measures bull/bear power via EMA13. Long when bull power > 0 with volume spike and 1w uptrend.
# Short when bear power < 0 with volume spike and 1w downtrend.
# Volume spike confirms institutional participation. 1w EMA50 filter ensures alignment with weekly trend.
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull (follows 1w uptrend longs) and bear (follows 1w downtrend shorts).

name = "6h_ElderRay_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w OHLC for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Elder Ray EMA13 (using prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d EMA13 for Elder Ray calculation
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d
    bear_power = low - ema_13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 2.0x 50-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(bull) or np.isnan(bear) or np.isnan(ema_trend):
            continue
            
        # Entry conditions
        # Long: bull power > 0 with volume spike and above 1w EMA50
        long_entry = (bull > 0) and vol_spike and (close[i] > ema_trend)
        # Short: bear power < 0 with volume spike and below 1w EMA50
        short_entry = (bear < 0) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power <= 0 or close below 1w EMA50
            if (bull <= 0) or (close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power >= 0 or close above 1w EMA50
            if (bear >= 0) or (close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals