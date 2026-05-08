#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_KeltnerBreakout_TrendVol"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once for pivot levels
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_high_w = np.roll(df_w['high'].values, 1)
    prev_low_w = np.roll(df_w['low'].values, 1)
    prev_close_w = np.roll(df_w['close'].values, 1)
    prev_high_w[0] = df_w['high'].values[0]
    prev_low_w[0] = df_w['low'].values[0]
    prev_close_w[0] = df_w['close'].values[0]
    
    # Weekly pivot point and S3/R3 levels
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3.0
    range_w = prev_high_w - prev_low_w
    s3_w = pivot_w - 2.0 * range_w  # S3
    r3_w = pivot_w + 2.0 * range_w  # R3
    
    # Align weekly levels to 6h
    s3_w_6h = align_htf_to_ltf(prices, df_w, s3_w)
    r3_w_6h = align_htf_to_ltf(prices, df_w, r3_w)
    
    # Get daily data for trend filter and Keltner channels
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Daily EMA20 for trend
    close_d = df_d['close'].values
    ema20_d = pd.Series(close_d).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_d = (close_d > ema20_d).astype(float)
    trend_d_6h = align_htf_to_ltf(prices, df_d, trend_d)
    
    # Daily ATR(10) for Keltner channels
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d_prev = np.roll(close_d, 1)
    close_d_prev[0] = close_d[0]
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - close_d_prev)
    tr3 = np.abs(low_d - close_d_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Keltner channels: EMA20 ± 2*ATR
    keltner_upper_d = ema20_d + 2.0 * atr10_d
    keltner_lower_d = ema20_d - 2.0 * atr10_d
    keltner_upper_6h = align_htf_to_ltf(prices, df_d, keltner_upper_d)
    keltner_lower_6h = align_htf_to_ltf(prices, df_d, keltner_lower_d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(s3_w_6h[i]) or np.isnan(r3_w_6h[i]) or 
            np.isnan(trend_d_6h[i]) or np.isnan(keltner_upper_6h[i]) or 
            np.isnan(keltner_lower_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3 with volume and daily uptrend
            long_cond = (close[i] > r3_w_6h[i] and vol_filter[i] and trend_d_6h[i] > 0.5)
            # Short: price breaks below weekly S3 with volume and daily downtrend
            short_cond = (close[i] < s3_w_6h[i] and vol_filter[i] and trend_d_6h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below daily Keltner lower (mean reversion)
            if close[i] < keltner_lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above daily Keltner upper (mean reversion)
            if close[i] > keltner_upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot S3/R3 breakouts with volume confirmation and daily trend filter on 6h.
# Uses weekly S3/R3 as strong support/resistance (2x range from pivot).
# Trend filter: daily price > EMA20 for long, < EMA20 for short.
# Exit: price re-enters daily Keltner channel (EMA20 ± 2*ATR) for mean reversion.
# Volume filter ensures institutional participation.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at S3/R3).
# Target: 20-40 trades/year to minimize fee decay while capturing significant moves.