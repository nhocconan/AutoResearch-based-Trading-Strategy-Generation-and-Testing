# 12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels on 12h chart act as strong support/resistance in both bull and bear markets.
# Breakouts above R3 or below S3 with volume confirmation and 1d trend filter capture momentum moves.
# Volume spike filter ensures breakouts have institutional participation.
# 1d EMA34 trend filter ensures we only trade in direction of higher timeframe trend.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets by catching breakouts to new highs.
# Works in bear markets by catching breakdowns to new lows.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_range = high_12h - low_12h
    r3_12h = close_12h + 1.1 * camarilla_range
    s3_12h = close_12h - 1.1 * camarilla_range
    
    # Align Camarilla levels to 1h timeframe (using close price as proxy for 1h alignment)
    # Since we're using 12h timeframe, we need to align to 12h bars
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on daily close
    ema_1d_34 = np.full(len(df_1d), np.nan)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1d)):
        if i < 33:
            ema_1d_34[i] = np.mean(close_1d[:i+1]) if i > 0 else close_1d[i]
        else:
            if np.isnan(ema_1d_34[i-1]):
                ema_1d_34[i] = np.mean(close_1d[i-33:i+1])
            else:
                ema_1d_34[i] = close_1d[i] * alpha + ema_1d_34[i-1] * (1 - alpha)
    
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Volume spike detection: current volume > 2 * 20-period average volume
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    volume_spike = np.full(n, False)
    volume_spike[20:] = volume[20:] > (2 * vol_ma_20[20:])
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup period
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(ema_1d_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and 1d uptrend
            if (price > r3_12h_aligned[i] and 
                volume_spike[i] and 
                ema_1d_34_aligned[i] > ema_1d_34_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and 1d downtrend
            elif (price < s3_12h_aligned[i] and 
                  volume_spike[i] and 
                  ema_1d_34_aligned[i] < ema_1d_34_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below S3 or 1d trend turns down
            if (price < s3_12h_aligned[i] or 
                ema_1d_34_aligned[i] < ema_1d_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or 1d trend turns up
            if (price > r3_12h_aligned[i] or 
                ema_1d_34_aligned[i] > ema_1d_34_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0