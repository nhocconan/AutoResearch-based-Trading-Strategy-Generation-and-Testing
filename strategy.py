#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Uses 1h timeframe with 4h/1d HTF for signal direction, targeting 60-150 total trades over 4 years (15-37/year)
# Session filter (08-20 UTC) reduces noise trades during low-volume periods
# Camarilla pivots from 1d provide institutional support/resistance levels
# 4h EMA50 ensures alignment with intermediate-term trend for higher probability trades
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts and bear markets via fade of false breakouts
# Discrete position sizing: 0.20 (20% of capital) balances exposure and risk while minimizing fee drag

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels (prior completed 1d bar's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least 2 days for prior day calculation
        return np.zeros(n)
    
    # Prior completed 1d bar's high/low/close for Camarilla calculation
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    # R3 = prior_close + (prior_high - prior_low) * 1.1/4
    # S3 = prior_close - (prior_high - prior_low) * 1.1/4
    # R4 = prior_close + (prior_high - prior_low) * 1.1/2
    # S4 = prior_close - (prior_high - prior_low) * 1.1/2
    range_1d = prior_high - prior_low
    r3 = prior_close + range_1d * 1.1 / 4
    s3 = prior_close - range_1d * 1.1 / 4
    r4 = prior_close + range_1d * 1.1 / 2
    s4 = prior_close - range_1d * 1.1 / 2
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close'].shift(1)).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 4h EMA50 (bullish trend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 AND price < 4h EMA50 (bearish trend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 4h EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 4h EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals