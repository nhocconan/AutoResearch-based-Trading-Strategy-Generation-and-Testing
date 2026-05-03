#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Uses Camarilla pivot levels from weekly data for institutional structure, EMA50 for trend filter,
# and volume spike for momentum confirmation. Designed to work in both bull and bear markets
# by taking breakouts in the direction of the higher timeframe trend. Target: 10-25 trades/year
# to minimize fee drag while capturing high-probability institutional level breaks.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Camarilla pivot levels from previous 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_vals = df_1w['close'].values
    
    # Pivot point
    pp = (high_1w + low_1w + close_1w_vals) / 3
    # Range
    rng = high_1w - low_1w
    # Camarilla levels
    r3 = pp + (rng * 1.1 / 4)
    s3 = pp - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (wait for 1w bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate volume regime: current 1d volume > 2.0x 20-period MA (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions
        # Long: break above R3 with volume spike and above 1w EMA50
        long_entry = (close[i] > r3_val) and vol_spike and (close[i] > ema_trend)
        # Short: break below S3 with volume spike and below 1w EMA50
        short_entry = (close[i] < s3_val) and vol_spike and (close[i] < ema_trend)
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below EMA50 (trend change)
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above EMA50 (trend change)
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals