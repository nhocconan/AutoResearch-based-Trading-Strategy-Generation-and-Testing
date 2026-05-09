#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d EMA34 trend filter and volume spike
# Works in bull markets (breakouts) and bear markets (reversions) by using Camarilla levels
# with trend filter to avoid counter-trend trades. Volume spike confirms institutional interest.
# Target: 20-50 trades/year to minimize fee drag.
name = "4h_Camarilla_R3_S3_1dEMA34_Trend_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Need previous day's high, low, close
    # Resample to daily using actual OHLC from 1d data (already daily)
    prev_close = df_1d['close'].values
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    
    # Calculate Camarilla levels for each day
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align to 4h - each daily level applies to all 4h bars of that day
    camarilla_r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 2.0x 24-period average volume (6 hours)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_4h[i]) or np.isnan(camarilla_r3_4h[i]) or np.isnan(camarilla_s3_4h[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        # Long: price crosses above S3 with volume spike and uptrend
        long_entry = (close[i] > camarilla_s3_4h[i] and 
                     close[i-1] <= camarilla_s3_4h[i] and  # crossed above
                     volume_spike[i] and 
                     close[i] > ema_34_4h[i])
        
        # Short: price crosses below R3 with volume spike and downtrend
        short_entry = (close[i] < camarilla_r3_4h[i] and 
                      close[i-1] >= camarilla_r3_4h[i] and  # crossed below
                      volume_spike[i] and 
                      close[i] < ema_34_4h[i])
        
        if position == 0:
            # Long: reversal from S3 support
            if long_entry:
                signals[i] = 0.25
                position = 1
            # Short: reversal from R3 resistance
            elif short_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below S3 or trend reversal
            if close[i] < camarilla_s3_4h[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above R3 or trend reversal
            if close[i] > camarilla_r3_4h[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals