#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation
# Camarilla pivot levels (R3, S3) provide strong intraday support/resistance
# 4h EMA50 filters for higher timeframe trend alignment
# 1d volume spike (>2.0x 20-bar average) confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in both bull and bear markets by combining mean reversion at extremes with trend filter

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume moving average for spike confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Camarilla pivot points for previous day (using daily data)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = PP + 1.1 * (High - Low) / 2
    # S3 = PP - 1.1 * (High - Low) / 2
    # where PP = (High + Low + Close) / 3
    
    # We need previous day's OHLC for today's Camarilla levels
    # Shift 1d data by 1 to get previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN as there's no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Calculate pivot point and Camarilla levels
    pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    r3 = pp + 1.1 * (prev_high_1d - prev_low_1d) / 2.0
    s3 = pp - 1.1 * (prev_high_1d - prev_low_1d) / 2.0
    
    # AlCamarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # max(50 for EMA, 20 for volume MA) + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema_4h = ema_50_4h_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_vol_ma_20_1d = vol_ma_20_1d_aligned[i]
        
        # Volume spike confirmation: current 1h volume > 2.0x 20-day average volume (scaled)
        # Approximate 1h volume expectation: 1d volume / 24
        vol_spike = curr_volume > 2.0 * (curr_vol_ma_20_1d / 24.0)
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price breaks below S3 (support) OR price below 4h EMA50 (trend change)
            if curr_low <= curr_s3 or curr_close < curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (resistance) OR price above 4h EMA50 (trend change)
            if curr_high >= curr_r3 or curr_close > curr_ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 (resistance) AND price above 4h EMA50 AND volume spike
            if (curr_high > curr_r3 and 
                curr_close > curr_ema_4h and
                vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S3 (support) AND price below 4h EMA50 AND volume spike
            elif (curr_low < curr_s3 and 
                  curr_close < curr_ema_4h and
                  vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals