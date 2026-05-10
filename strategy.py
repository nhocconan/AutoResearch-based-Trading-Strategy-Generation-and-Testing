#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Filtered
# Hypothesis: Uses 12h timeframe with 1d timeframe for higher trend confirmation.
# Enters long when price breaks above daily R3 in uptrend (close > EMA50) with volume > 4x 20-period average.
# Enters short when price breaks below daily S3 in downtrend (close < EMA50) with volume confirmation.
# Exits when price returns to opposite level (S3 for long, R3 for short) or trend reverses.
# Uses daily EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25 to minimize fee drag.
# IMPROVEMENTS: Added minimum holding period of 3 bars to reduce churn, tightened volume filter to 4x average, added ADX filter for trend strength.

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Filtered"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx[np.isnan(dx) | (plus_di + minus_di == 0)] = 0
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r3_level = prev_close + 1.1 * pivot_range
    s3_level = prev_close - 1.1 * pivot_range
    
    # Align pivot levels to 12h timeframe (available after 1d bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume filter: volume > 4x 20-period average on 12h chart (tighter filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    start_idx = max(60, 20)  # Warmup for EMA, ADX and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Trend filter: price above/below 1d EMA50 and ADX > 25 for trend strength
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Require minimum 3 bars since last exit to prevent churn
            if bars_since_entry >= 3:
                # Long entry: price breaks above R3 in uptrend with volume spike and strong trend
                if (close[i] > r3_aligned[i] and 
                    price_above_ema and 
                    strong_trend and
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short entry: price breaks below S3 in downtrend with volume spike and strong trend
                elif (close[i] < s3_aligned[i] and 
                      price_below_ema and 
                      strong_trend and
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Long exit: price returns to S3 or trend reverses to downtrend or weak trend
            if (close[i] < s3_aligned[i] or 
                price_below_ema or 
                not strong_trend):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                bars_since_entry += 1
        elif position == -1:
            # Short exit: price returns to R3 or trend reverses to uptrend or weak trend
            if (close[i] > r3_aligned[i] or 
                price_above_ema or 
                not strong_trend):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                bars_since_entry += 1
    
    return signals