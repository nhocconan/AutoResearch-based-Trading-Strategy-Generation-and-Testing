#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeRegime
Hypothesis: On daily timeframe, Camarilla R3/S3 breakouts with weekly trend filter (EMA34) and volume confirmation (top 30%) capture strong momentum moves. Weekly trend ensures alignment with higher timeframe direction, reducing false breakouts. Volume confirms participation. Fixed size 0.25 limits trades to target 30-100 over 4 years. Works in bull via breakouts, in bear via short breakdowns with trend filter.
"""

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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load daily data for Camarilla levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to daily timeframe (no shift needed as 1d->1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for volume percentile, 34 for weekly EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        weekly_trend_up = close_val > ema_34_1w_aligned[i]
        weekly_trend_down = close_val < ema_34_1w_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume regime AND weekly trend alignment
        long_entry = (close_val > camarilla_r3_val) and vol_regime and weekly_trend_up
        short_entry = (close_val < camarilla_s3_val) and vol_regime and weekly_trend_down
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center) or trend change
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_changed = close_val < ema_34_1w_aligned[i]  # weekly trend turned down
            if close_val < mid_point or trend_changed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center) or trend change
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_changed = close_val > ema_34_1w_aligned[i]  # weekly trend turned up
            if close_val > mid_point or trend_changed:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeRegime"
timeframe = "1d"
leverage = 1.0