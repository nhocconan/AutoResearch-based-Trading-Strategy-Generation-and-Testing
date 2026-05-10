#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Redux
# Hypothesis: Reduced trade frequency via stricter conditions (volume > 4x avg, 3-bar hold minimum)
# and added 1d ADX trend filter to avoid chop. Targets 15-25 trades/year for 12h timeframe.
# Uses price > 1d EMA50 for uptrend, price < 1d EMA50 for downtrend to avoid whipsaws.
# Position size 0.25 to balance return and drawdown.

name = "12H_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Redux"
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
    
    # Get 1d data for Camarilla pivots, EMA trend, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(np.abs(high_1d[1:] - low_1d[:-1]), np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    atr_1d = np.zeros_like(tr)
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14 if i >= 1 else tr[i]
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / np.where(atr_1d == 0, 1, atr_1d)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / np.where(atr_1d == 0, 1, atr_1d)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r3_level = prev_close + 1.1 * pivot_range
    s3_level = prev_close - 1.1 * pivot_range
    
    # Align pivot levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_level)
    
    # Volume filter: volume > 4x 20-period average on 12h chart (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 4.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    start_idx = max(60, 20)  # Warmup for EMA, ADX, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Trend filters: price above/below 1d EMA50 AND ADX > 20 (trending market)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        trending = adx_1d_aligned[i] > 20
        
        if position == 0:
            # Require minimum 3 bars since last exit to prevent churn
            if bars_since_entry >= 3:
                # Long entry: price breaks above R3 in uptrend with volume spike
                if (close[i] > r3_aligned[i] and 
                    price_above_ema and 
                    trending and 
                    volume[i] > vol_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                # Short entry: price breaks below S3 in downtrend with volume spike
                elif (close[i] < s3_aligned[i] and 
                      price_below_ema and 
                      trending and 
                      volume[i] > vol_threshold[i]):
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        elif position == 1:
            # Long exit: price returns to S3 or trend reverses to downtrend
            if (close[i] < s3_aligned[i] or 
                not price_above_ema or 
                not trending):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
                bars_since_entry += 1
        elif position == -1:
            # Short exit: price returns to R3 or trend reverses to uptrend
            if (close[i] > r3_aligned[i] or 
                not price_below_ema or 
                not trending):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
                bars_since_entry += 1
    
    return signals