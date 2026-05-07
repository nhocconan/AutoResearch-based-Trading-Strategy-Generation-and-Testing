#!/usr/bin/env python3
"""
1h_Camarilla_R3S3_4hTrend_Volume_1dVolFilter
Hypothesis: Use Camarilla pivot levels from 4h timeframe for entry signals (R3/S3 breakout), 
filtered by 4h trend (EMA50) and daily volume filter to avoid false signals. 
Long when price breaks above R3 with 4h uptrend and daily volume spike. 
Short when price breaks below S3 with 4h downtrend and daily volume spike.
Camarilla provides precise support/resistance levels, and volume filter ensures 
institutional participation. Works in both bull and bear markets by requiring 
alignment with 4h trend and volume confirmation.
Timeframe: 1h, using 4h for signal direction and 1h for entry timing.
Target: 15-35 trades/year per symbol to stay under fee drag limits.
"""
name = "1h_Camarilla_R3S3_4hTrend_Volume_1dVolFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 4h timeframe
    # Using previous 4h bar's OHLC (standard Camarilla)
    ph = df_4h['high'].shift(1).values  # Previous 4h high
    pl = df_4h['low'].shift(1).values   # Previous 4h low
    pc = df_4h['close'].shift(1).values # Previous 4h close
    
    # Handle first value where shift creates NaN
    ph[0] = ph[1] if len(ph) > 1 else high[0]
    pl[0] = pl[1] if len(pl) > 1 else low[0]
    pc[0] = pc[1] if len(pc) > 1 else close[0]
    
    # Camarilla R3 and S3 levels
    r3 = pc + 1.1 * (ph - pl) / 2
    s3 = pc - 1.1 * (ph - pl) / 2
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Daily volume filter: current volume > 1.5 * 20-day average volume
    vol_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    volume_filter = df_1d['volume'].values > (vol_20 * 1.5)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup for all indicators
    start_idx = max(50, 20)  # EMA50 and 20-day volume average
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 4h uptrend (price > EMA50) + daily volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 + 4h downtrend (price < EMA50) + daily volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        else:  # position == -1
            # Short exit: price breaks above R3 (reversal signal)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals