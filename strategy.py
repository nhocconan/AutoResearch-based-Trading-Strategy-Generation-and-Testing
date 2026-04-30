#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3/S3) from daily structure for institutional breakout levels
# Only trade breakouts above R3 or below S3 in direction of 12h EMA50 trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous day
    # Typical Camarilla uses previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    # Simplified: range = high - low, R3 = close + 1.1*range*1.1/4, S3 = close - 1.1*range*1.1/4
    # Further simplified: R3 = close + 1.1*(high-low)*0.275, S3 = close - 1.1*(high-low)*0.275
    # Actually Camarilla R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_multiplier = 1.1 / 4  # 0.275
    r3 = close_1d + camarilla_multiplier * (high_1d - low_1d)
    s3 = close_1d - camarilla_multiplier * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 50, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND above 12h EMA50 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 12h EMA50 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 12h EMA50 (trend change)
            if curr_close < curr_s3 or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 12h EMA50 (trend change)
            if curr_close > curr_r3 or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals