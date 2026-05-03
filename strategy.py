#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND 12h EMA50 rising AND volume > 2.0x 20-period MA
# Short: Close breaks below Camarilla S3 AND 12h EMA50 falling AND volume > 2.0x 20-period MA
# Exit: Opposite Camarilla breakout or EMA50 flattens or volume drops.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla provides precise intraday support/resistance; 12h EMA50 filters for strong trends only;
# high volume threshold (2.0x) reduces false breakouts significantly. Works in bull via longs and
# bear via shorts when aligned with 12h trend. Prioritizes trade quality over frequency to minimize fee drag.

name = "4h_Camarilla_R3S3_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h EMA50 slope (rising/falling)
    ema_slope = np.diff(ema_50_12h_aligned, prepend=ema_50_12h_aligned[0])
    ema_rising = ema_slope > 0
    ema_falling = ema_slope < 0
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    # Need 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla: based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # True range for Camarilla calculation
    tr = prev_high - prev_low
    
    # Camarilla levels
    camarilla_r3 = prev_close + tr * 1.1 / 4
    camarilla_s3 = prev_close - tr * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels for current day)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: current 4h volume > 2.0x 20-period MA (strict threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        ema_rising_val = ema_rising[i]
        ema_falling_val = ema_falling[i]
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND EMA50 rising AND volume spike
            if close_val > camarilla_r3_val and ema_rising_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND EMA50 falling AND volume spike
            elif close_val < camarilla_s3_val and ema_falling_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR EMA50 falling OR volume drops
            if close_val < camarilla_s3_val or ema_falling_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR EMA50 rising OR volume drops
            if close_val > camarilla_r3_val or ema_rising_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals