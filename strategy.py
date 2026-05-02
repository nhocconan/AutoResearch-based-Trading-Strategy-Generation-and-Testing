#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Uses 12h timeframe to target 50-150 trades over 4 years (12-37/year) to minimize fee drag
# Camarilla pivots from 1w provide major institutional support/resistance levels
# 1w EMA50 ensures alignment with weekly trend for higher probability trades
# Volume spike (2x 30-period average) confirms institutional participation
# Works in bull markets via breakouts and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Camarilla levels (prior completed 1w bar's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for prior week calculation
        return np.zeros(n)
    
    # Prior completed 1w bar's high/low/close for Camarilla calculation
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3, R4, S4
    # R3 = prior_close + (prior_high - prior_low) * 1.1/4
    # S3 = prior_close - (prior_high - prior_low) * 1.1/4
    # R4 = prior_close + (prior_high - prior_low) * 1.1/2
    # S4 = prior_close - (prior_high - prior_low) * 1.1/2
    range_1w = prior_high - prior_low
    r3 = prior_close + range_1w * 1.1 / 4
    s3 = prior_close - range_1w * 1.1 / 4
    r4 = prior_close + range_1w * 1.1 / 2
    s4 = prior_close - range_1w * 1.1 / 2
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].shift(1)).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume spike (2x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1w EMA50 (bullish trend) AND volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND price < 1w EMA50 (bearish trend) AND volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below S3 OR below 1w EMA50 (trend change)
            if close[i] < s3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above R3 OR above 1w EMA50 (trend change)
            if close[i] > r3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals