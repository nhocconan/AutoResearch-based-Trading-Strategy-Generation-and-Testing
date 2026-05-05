#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d volume spike filter and session timing (08-20 UTC)
# Long when price breaks above 4h Camarilla R3 AND volume > 2.0 * avg_volume(20) on 1d AND in session (08-20 UTC)
# Short when price breaks below 4h Camarilla S3 AND volume > 2.0 * avg_volume(20) on 1d AND in session (08-20 UTC)
# Exit when price crosses back through the 4h Camarilla midpoint (R3/S3 average)
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# 4h Camarilla provides structure from higher timeframe, 1d volume spike confirms institutional interest
# Session filter reduces noise during low-liquidity hours (20-08 UTC)

name = "1h_4hCamarillaR3S3_1dVolumeSpike_Session"
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
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (R3, S3, midpoint)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    high_low_4h = high_4h - low_4h
    camarilla_r3_4h = close_4h + 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_s3_4h = close_4h - 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_mid_4h = (camarilla_r3_4h + camarilla_s3_4h) / 2.0
    
    # Align 4h Camarilla to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid_4h)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for volume average
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (20-period)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Align 1d volume spike to 1h timeframe (wait for completed 1d bar)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla R3, 1d volume spike, in session
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla S3, 1d volume spike, in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above 4h Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals