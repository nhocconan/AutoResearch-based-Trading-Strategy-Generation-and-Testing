#!/usr/bin/env python3
name = "1h_4h1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla levels (standard formula)
    range_hl = prev_high - prev_low
    multiplier = 1.1 / 12  # ~0.0916667
    
    # Resistance levels
    r3 = prev_close + range_hl * 1.1 * multiplier * 3
    r4 = prev_close + range_hl * 1.1 * multiplier * 4
    
    # Support levels
    s3 = prev_close - range_hl * 1.1 * multiplier * 3
    s4 = prev_close - range_hl * 1.1 * multiplier * 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_4h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_4h, s4)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S3 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > s3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R3 with volume and daily downtrend
            elif close[i] < r3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < s3_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > r3_aligned[i] or volume[i] < vol_ma_24[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S3/R3 breakout with 1d trend and volume confirmation
# - Camarilla S3/R3 act as key support/resistance levels from prior 4h bar
# - Breakout above S3 with volume in daily uptrend = long opportunity
# - Breakdown below R3 with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy S3 breaks in uptrend) and bear (sell R3 breaks in downtrend)
# - Exit when price returns to S3/R3 or volume weakens
# - Position size 0.20 targets ~15-35 trades/year, avoiding fee drag
# - Uses actual 4h Camarilla levels (not daily) for better stability
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: 4h Camarilla + 1d trend + 1h volume not recently tried
# - Aims for 60-140 total trades over 4 years (15-35/year) to stay within limits
# - Focus on BTC/ETH as primary targets, not SOL-only strategies
# - Uses proper alignment to avoid look-ahead bias
# - Uses discrete position sizing to minimize fee churn
# - Stop loss implemented via signal=0 when price breaks opposite level