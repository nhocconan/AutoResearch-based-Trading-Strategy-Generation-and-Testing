#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w trend filter + volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance from prior 1d session
# Breakout of R3 (resistance 3) or S3 (support 3) with volume confirms institutional participation
# 1w EMA50 trend filter ensures we trade with the weekly momentum
# Discrete position sizing 0.25 minimizes fee churn while maintaining adequate exposure
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by adapting to weekly trend direction

name = "12h_Camarilla_R3S3_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla levels: 
    # R4 = close + ((high - low) * 1.5/2)
    # R3 = close + ((high - low) * 1.25/2)
    # R2 = close + ((high - low) * 1.1/2)
    # R1 = close + ((high - low) * 0.5/2)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 0.5/2)
    # S2 = close - ((high - low) * 1.1/2)
    # S3 = close - ((high - low) * 1.25/2)
    # S4 = close - ((high - low) * 1.5/2)
    
    # We need prior day's OHLC, so shift by 1
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    
    # First day has no prior day
    prev_high[0] = prev_low[0] = prev_close[0] = 0
    
    # Calculate Camarilla levels
    hl_range = prev_high - prev_low
    r3 = prev_close + (hl_range * 1.25 / 2)
    s3 = prev_close - (hl_range * 1.25 / 2)
    
    # Align 1d Camarilla levels to 12h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 12h volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for pivots, EMA and volume MA)
    start_idx = 50  # max(20 for volume, 50 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # In uptrend: long breakout of R3 with volume
                if (close[i] > r3_aligned[i] and 
                    i > start_idx and close[i-1] <= r3_aligned[i-1] and
                    volume_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # In downtrend: short breakdown of S3 with volume
                if (close[i] < s3_aligned[i] and 
                    i > start_idx and close[i-1] >= s3_aligned[i-1] and
                    volume_confirm[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price breaks below R3 (failed breakout) or reverse signal
            if close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price breaks above S3 (failed breakdown) or reverse signal
            if close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals