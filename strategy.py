# 1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
# 1d strategy using weekly trend filter with Camarilla breakout and volume confirmation
# Targets 20-50 trades/year to minimize fee drag. Works in both bull and bear by trading with weekly trend.
# Entry: Price breaks above R3 (bull) or below S3 (bear) with volume spike and aligned with weekly trend
# Exit: Price breaks back below S3 (long) or above R3 (short) or crosses weekly trend

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1w['high'].values  # Use weekly high/low for ATR calculation
    low_1d = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1w[:-1]), 
                                np.abs(low_1d[1:] - close_1w[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1w, atr14)
    
    # Calculate daily Camarilla levels from previous daily bar
    # Camarilla formula: range = high - low
    # R3 = close + range * 1.1/4, S3 = close - range * 1.1/4
    range_1d = high - low
    camarilla_r3 = close + range_1d * 1.1 / 4
    camarilla_s3 = close - range_1d * 1.1 / 4
    
    # Shift to get previous day's levels (no look-ahead)
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(camarilla_r3_prev[i]) or np.isnan(camarilla_s3_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above weekly EMA
            if (close[i] > camarilla_r3_prev[i] and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below weekly EMA
            elif (close[i] < camarilla_s3_prev[i] and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below weekly EMA
            if (close[i] < camarilla_s3_prev[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above weekly EMA
            if (close[i] > camarilla_r3_prev[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals