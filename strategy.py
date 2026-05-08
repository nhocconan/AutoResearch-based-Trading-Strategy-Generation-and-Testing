#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Long when price pulls back to S3 level during 1d uptrend with volume confirmation.
# Short when price rallies to R3 level during 1d downtrend with volume confirmation.
# Exit when price reaches opposite S1/R1 level or trend fails.
# Camarilla levels provide statistically significant support/resistance; 1d trend filters direction; volume confirms reversal strength.
# Designed to capture mean-reversion moves within established trends, working in both bull and bear markets.
# Target: 20-40 trades/year to stay within profitable range.

name = "6h_Camarilla_R3_S3_Reversal_1dTrend_Volume"
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
    
    # Get 60-period 6h data for Camarilla calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 60:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 60-period high, low, close for Camarilla levels
    hh = np.max(high_6h[-60:])
    ll = np.min(low_6h[-60:])
    c = close_6h[-1]
    
    # Calculate Camarilla levels for current period
    range_val = hh - ll
    camarilla_s3 = c - (range_val * 1.1 / 6)
    camarilla_s1 = c - (range_val * 1.1 / 12)
    camarilla_r1 = c + (range_val * 1.1 / 12)
    camarilla_r3 = c + (range_val * 1.1 / 6)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA20 for trend filter
    ema_20 = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).values
    ema_20_prev = np.roll(ema_20, 1)
    ema_20_prev[0] = ema_20[0]
    ema_20_rising = ema_20 > ema_20_prev
    ema_20_falling = ema_20 < ema_20_prev
    
    # Get 6h 20-period average volume for volume filter
    vol_ma_20 = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, np.full(len(df_6h), camarilla_s3))
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_6h, np.full(len(df_6h), camarilla_s1))
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_6h, np.full(len(df_6h), camarilla_r1))
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, np.full(len(df_6h), camarilla_r3))
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    ema_20_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_20_rising)
    ema_20_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_20_falling)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_20_rising_aligned[i]) or \
           np.isnan(ema_20_falling_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 6h bar's volume
            idx_6h = 0
            while idx_6h < len(df_6h) and df_6h.iloc[idx_6h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_6h += 1
            idx_6h -= 1  # last completed 6h bar
            
            if idx_6h >= 0:
                vol_6h_current = df_6h.iloc[idx_6h]['volume']
                vol_filter = vol_6h_current > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla S3/R3 reversal with trend and volume
            # Long when price touches S3 level during 1d uptrend with volume spike
            long_condition = (low[i] <= camarilla_s3_aligned[i]) and \
                             ema_20_rising_aligned[i] and vol_filter
            # Short when price touches R3 level during 1d downtrend with volume spike
            short_condition = (high[i] >= camarilla_r3_aligned[i]) and \
                              ema_20_falling_aligned[i] and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches S1 level or trend fails
            if (high[i] >= camarilla_s1_aligned[i]) or (not ema_20_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches R1 level or trend fails
            if (low[i] <= camarilla_r1_aligned[i]) or (not ema_20_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals