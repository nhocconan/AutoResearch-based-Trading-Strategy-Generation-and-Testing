#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot (R3/S3) Breakout with 1w EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from 1d for structure, breakout of R3/S3 for entry
# 1w EMA34 filter ensures trades only in direction of weekly trend
# Volume confirmation (2.0x 20-period average) filters low-quality breakouts
# Works in bull markets via buying R3 breakouts in uptrends and bear markets via selling S3 breakdowns in downtrends
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Camarilla_R3_S3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels for 1d
    # Based on previous day's OHLC
    # R4 = close + ((high - low) * 1.5)
    # R3 = close + ((high - low) * 1.25)
    # S3 = close - ((high - low) * 1.25)
    # S4 = close - ((high - low) * 1.5)
    # Using 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    r3 = prev_close + (prev_range * 1.25)
    s3 = prev_close - (prev_range * 1.25)
    
    # Align to 1d timeframe (no additional delay needed as these are based on completed previous day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34, 20)  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1w = ema_34_1w_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above R3 AND above 1w EMA34 (uptrend)
                if curr_close > curr_r3 and curr_close > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below S3 AND below 1w EMA34 (downtrend)
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price falls below S3 or below 1w EMA34
            if curr_close < curr_s3 or curr_close < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above R3 or above 1w EMA34
            if curr_close > curr_r3 or curr_close > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals