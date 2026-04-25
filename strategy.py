#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrendFilter_VolumeSpike_v3
Hypothesis: Trade 4h Camarilla R3/S3 breakouts aligned with daily EMA34 trend and volume spike (>2.0*ATR14).
Tighter volume threshold (2.0 vs 1.8) and minimum holding period (6 bars) to reduce overtrading vs prior variants.
Only trade in direction of daily trend to avoid whipsaws. Uses discrete sizing 0.25 to limit fee drag.
Target: 10-25 trades/year to avoid fee drag while maintaining edge. Works in bull/bear via daily trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR14 for volume confirmation
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate previous day's Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_pp = (high_1d + low_1d + close_1d) / 3
    camarilla_range = high_1d - low_1d
    camarilla_r3 = camarilla_pp + camarilla_range * 1.1 / 4
    camarilla_s3 = camarilla_pp - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start index: need warmup for daily EMA34, ATR, and previous day's data
    start_idx = max(34, 14) + 1  # +1 because we use previous day's data
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 2.0 * ATR (tighter to reduce trades)
        volume_confirm = volume[i] > 2.0 * atr[i]
        
        # Determine daily trend from EMA34
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)[i]
        if np.isnan(daily_close_aligned):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        if daily_close_aligned > ema_34_1d_aligned[i]:
            daily_trend = 'bullish'  # only allow longs
        elif daily_close_aligned < ema_34_1d_aligned[i]:
            daily_trend = 'bearish'  # only allow shorts
        else:
            daily_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            bars_since_entry = 0
            # Long setup: price breaks above Camarilla R3 AND volume confirm AND bullish daily trend
            long_setup = (close[i] > camarilla_r3_aligned[i]) and volume_confirm and (daily_trend == 'bullish')
            
            # Short setup: price breaks below Camarilla S3 AND volume confirm AND bearish daily trend
            short_setup = (close[i] < camarilla_s3_aligned[i]) and volume_confirm and (daily_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Minimum holding period: 6 bars (~24 hours for 4h)
            if bars_since_entry < 6:
                signals[i] = 0.25
            else:
                # Long: hold position
                signals[i] = 0.25
                # Exit: price breaks below Camarilla S3 OR daily trend turns bearish
                if (close[i] < camarilla_s3_aligned[i]) or (daily_trend == 'bearish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 6 bars (~24 hours for 4h)
            if bars_since_entry < 6:
                signals[i] = -0.25
            else:
                # Short: hold position
                signals[i] = -0.25
                # Exit: price breaks above Camarilla R3 OR daily trend turns bullish
                if (close[i] > camarilla_r3_aligned[i]) or (daily_trend == 'bullish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrendFilter_VolumeSpike_v3"
timeframe = "4h"
leverage = 1.0