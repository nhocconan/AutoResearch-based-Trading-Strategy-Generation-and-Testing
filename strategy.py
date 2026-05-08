#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    range_4h = high - low
    camarilla_r3 = close + range_4h * 1.1 / 4
    camarilla_s3 = close - range_4h * 1.1 / 4
    camarilla_r4 = close + range_4h * 1.1 / 2
    camarilla_s4 = close - range_4h * 1.1 / 2
    
    # Shift to get previous bar's levels (no look-ahead)
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
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_prev[i]) or 
            np.isnan(camarilla_s3_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 1d EMA
            if (close[i] > camarilla_r3_prev[i] and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below S3 with volume spike, below 1d EMA
            elif (close[i] < camarilla_s3_prev[i] and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below 1d EMA
            if (close[i] < camarilla_s3_prev[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R3 OR above 1d EMA
            if (close[i] > camarilla_r3_prev[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Uses 4h Camarilla R3/S3 breakouts with volume confirmation and 1d EMA trend filter.
# - Enters long when price breaks above R3 (previous bar) with volume spike and above 1d EMA
# - Enters short when price breaks below S3 (previous bar) with volume spike and below 1d EMA
# - Exits when price breaks back below S3 (long) or above R3 (short) OR crosses 1d EMA
# - Volume spike filter ensures breakouts have conviction
# - 1d EMA filter ensures trading with higher timeframe trend
# - Camarilla levels provide natural support/resistance at key levels
# - Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag
# - Position size: 0.30 for balanced risk/return
# - Works in both bull and bear markets by following 1d trend direction
# - Volume confirmation reduces false breakouts in low-volume environments
# - Focus on BTC and ETH as primary targets (not SOL-only)