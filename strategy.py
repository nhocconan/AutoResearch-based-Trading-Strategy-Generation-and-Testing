#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout at Camarilla R3/S3 levels on 4h, filtered by 1d EMA34 trend and volume spike (>2x average). Uses ATR-based stoploss.
# Designed for 20-50 trades/year to avoid fee drag. Works in bull/bear via trend filter.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Use previous day's high, low, close
    prev_day_high = np.full(n, np.nan)
    prev_day_low = np.full(n, np.nan)
    prev_day_close = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_day_high[i] = df_1d['high'].values[i-1] if i-1 < len(df_1d) else np.nan
        prev_day_low[i] = df_1d['low'].values[i-1] if i-1 < len(df_1d) else np.nan
        prev_day_close[i] = df_1d['close'].values[i-1] if i-1 < len(df_1d) else np.nan
    
    # Align to 4h timeframe
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    prev_day_close_aligned = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(prev_day_high_aligned[i]) or np.isnan(prev_day_low_aligned[i]) or np.isnan(prev_day_close_aligned[i])):
            range_val = prev_day_high_aligned[i] - prev_day_low_aligned[i]
            camarilla_r3[i] = prev_day_close_aligned[i] + range_val * 1.1 / 4
            camarilla_s3[i] = prev_day_close_aligned[i] - range_val * 1.1 / 4
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R3 with volume spike
                if close[i] > camarilla_r3[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S3 with volume spike
                if close[i] < camarilla_s3[i] and volume[i] > 2.0 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA34 or stoploss hit
            if close[i] < ema_34_1d_aligned[i] or (i > 0 and low[i] < camarilla_s3[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA34 or stoploss hit
            if close[i] > ema_34_1d_aligned[i] or (i > 0 and high[i] > camarilla_r3[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals