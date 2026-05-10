#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R3S3_Breakout_12hEMA50_Trend_Volume
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Works in bull/bear markets by trading breakouts with trend alignment. Low trade frequency via
# strict breakout conditions (only at specific pivot levels) and volume filter.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_Camarilla_Pivot_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # Daily data for Camarilla pivot levels (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift to get previous day's values (avoid look-ahead)
    prev_close = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: current volume > 1.3 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.3
        
        if position == 0:
            # Long: price breaks above Camarilla R3 with uptrend and volume
            if (close[i] > camarilla_r3_aligned[i] and
                trend_12h_up_aligned[i] > 0.5 and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 with downtrend and volume
            elif (close[i] < camarilla_s3_aligned[i] and
                  trend_12h_down_aligned[i] > 0.5 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Camarilla pivot (central level) or trend fails
            # Calculate central pivot: (H+L+C)/3
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] < camarilla_pivot_aligned or
                trend_12h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Camarilla pivot or trend fails
            camarilla_pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3
            camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full_like(prev_close, camarilla_pivot))[i] if not np.isnan(prev_high[i]) else camarilla_pivot
            
            if (close[i] > camarilla_pivot_aligned or
                trend_12h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals