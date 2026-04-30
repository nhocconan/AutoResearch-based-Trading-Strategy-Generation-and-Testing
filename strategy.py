#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA21 trend filter and volume confirmation.
# Uses Camarilla pivot levels (R3/S3) for breakout entries, 4h EMA21 for trend direction,
# and volume >1.5x 20-bar average for confirmation. Session filter (08-20 UTC) reduces noise.
# Discrete position sizing at ±0.20 to manage fee drag on 1h timeframe.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid excessive fees.
# Works in bull markets via breakout continuation and in bear markets via volatility expansion capture.

name = "1h_Camarilla_R3S3_Breakout_4hEMA21_Trend_VolumeConfirm_Session_v1"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend filter
    close_4h_vals = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h_vals).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMA21 to 1h timeframe
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate daily Camarilla pivots (based on previous day's OHLC)
    # We need daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (use previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_21_4h = ema_21_4h_aligned[i]
        curr_r3 = camarilla_r3_aligned[i]
        curr_s3 = camarilla_s3_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above R3, above 4h EMA21, with volume confirmation
            if (curr_close > curr_r3 and 
                curr_close > curr_ema_21_4h and 
                curr_volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3, below 4h EMA21, with volume confirmation
            elif (curr_close < curr_s3 and 
                  curr_close < curr_ema_21_4h and 
                  curr_volume_confirm):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price returns to midpoint between R3 and S3 (mean reversion)
            midpoint = (curr_r3 + curr_s3) / 2
            if curr_close < midpoint:  # Price back below midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit condition: price returns to midpoint between R3 and S3
            midpoint = (curr_r3 + curr_s3) / 2
            if curr_close > midpoint:  # Price back above midpoint
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals