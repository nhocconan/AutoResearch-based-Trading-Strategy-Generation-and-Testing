#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike confirmation.
# R3/S3 are stronger support/resistance levels than S1/R1, reducing false breakouts.
# EMA34 ensures alignment with higher-timeframe trend to avoid counter-trend trades.
# Volume spike (>2x average) confirms institutional interest at key levels.
# Works in bull markets (breaking R3 in uptrend) and bear markets (breaking S3 in downtrend).
# Target: 20-50 trades/year with disciplined entries to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (S3, R3)
    P = (high_1d + low_1d + close_1d) / 3
    R3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Get 12h data for EMA34 trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA34 for trend filter
    ema34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_20[i] = np.mean(volume_12h[i-20:i])
    
    # Align all indicators to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema34_aligned = align_htf_to_ltf(prices, df_12h, ema34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 2.0x 20-period average
        vol_filter = False
        if not np.isnan(vol_ma_20_aligned[i]):
            # Find current 12h bar's volume
            idx_12h = 0
            while idx_12h < len(df_12h) and df_12h.iloc[idx_12h]['open_time'] <= prices.iloc[i]['open_time']:
                idx_12h += 1
            idx_12h -= 1  # last completed 12h bar
            
            if idx_12h >= 0:
                vol_12h_current = df_12h.iloc[idx_12h]['volume']
                vol_filter = vol_12h_current > 2.0 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout + trend + volume spike
            # Long when price breaks above R3 in uptrend (price > EMA34) with volume spike
            long_condition = (close[i] > R3_aligned[i] * 1.001) and \
                             (close[i] > ema34_aligned[i]) and vol_filter
            # Short when price breaks below S3 in downtrend (price < EMA34) with volume spike
            short_condition = (close[i] < S3_aligned[i] * 0.999) and \
                              (close[i] < ema34_aligned[i]) and vol_filter
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 or trend fails
            if (close[i] < S3_aligned[i]) or (close[i] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 or trend fails
            if (close[i] > R3_aligned[i]) or (close[i] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals