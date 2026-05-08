# 138451: 6h_Camarilla_Pivot_Reversal_with_1d_Trend_and_Volume
# Hypothesis: Camarilla pivot reversals at R3/S3 (fade) and R4/S4 (breakout) on 6h timeframe work in both bull and bear markets when filtered by 1d trend and volume spikes.
# In bull markets, R4 breakouts capture continuation; in bear markets, R3 fades capture mean reversion. Volume confirms institutional participation.
# Timeframe: 6h, HTF: 1d for trend filter.
# Target: 50-150 trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_Pivot_Reversal_with_1d_Trend_and_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.500
    # R3 = C + (H-L) * 1.250
    # S3 = C - (H-L) * 1.250
    # S4 = C - (H-L) * 1.500
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    rng = daily_high - daily_low
    r4 = daily_close + rng * 1.500
    r3 = daily_close + rng * 1.250
    s3 = daily_close - rng * 1.250
    s4 = daily_close - rng * 1.500
    
    # Align daily pivot levels to 6h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Daily EMA34 for trend filter (only needs completed daily candle)
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions:
            # 1. Breakout above R4 with volume spike (continuation)
            # 2. Reversal from S3 with volume spike (mean reversion in uptrend)
            long_breakout = (close[i] > r4_aligned[i] and volume_spike[i])
            long_reversal = (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                           ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_spike[i])
            
            # Short conditions:
            # 1. Breakdown below S4 with volume spike (continuation)
            # 2. Reversal from R3 with volume spike (mean reversion in downtrend)
            short_breakdown = (close[i] < s4_aligned[i] and volume_spike[i])
            short_reversal = (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                            ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_spike[i])
            
            if long_breakout or long_reversal:
                signals[i] = 0.25
                position = 1
            elif short_breakdown or short_reversal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below R3 (failure of bullish momentum) or above R4 then fails
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above S3 (failure of bearish momentum) or below S4 then fails
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals