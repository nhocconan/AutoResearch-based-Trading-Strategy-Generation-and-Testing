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
    
    # Calculate daily Camarilla pivot levels (standard formula)
    # P = (H + L + C) / 3
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    # Using previous day's H, L, C
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    P = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    HL = high_1d_prev - low_1d_prev
    R3 = close_1d_prev + HL * 1.1 / 4.0
    S3 = close_1d_prev - HL * 1.1 / 4.0
    
    # Align daily pivot levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema34_1d_aligned[i]
        r3_val = R3_aligned[i]
        s3_val = S3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 1d EMA
            if (close[i] > r3_val and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below S3 with volume spike, below 1d EMA
            elif (close[i] < s3_val and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below 1d EMA
            if (close[i] < s3_val or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R3 OR above 1d EMA
            if (close[i] > r3_val or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Uses 4-hour Camarilla R3/S3 breakouts with volume confirmation and 1d EMA trend filter.
# - Enters long when price breaks above R3 level with volume spike and above 1d EMA
# - Enters short when price breaks below S3 level with volume spike and below 1d EMA
# - Exits when price breaks back below S3 (long) or above R3 (short) OR crosses 1d EMA
# - Volume spike filter ensures breakouts have conviction
# - 1d EMA filter ensures trading with higher timeframe trend
# - Camarilla levels provide clear support/resistance for breakout trading
# - Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag
# - Position size: 0.30 for balanced risk/return
# - Works in both bull and bear markets by following 1d trend direction
# - Volume confirmation reduces false breakouts in low-volume environments
# - Focus on BTC and ETH as primary targets (not SOL-only)