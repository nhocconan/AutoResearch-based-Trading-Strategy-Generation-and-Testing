#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_RegimeFilter_v1
Hypothesis: Trade 4h breakouts from Camarilla R3/S3 levels with 12h EMA50 trend filter and chop regime filter. Uses 12h HTF for trend alignment and choppiness index to avoid whipsaws in ranging markets. Discrete size 0.30 targets 20-40 trades/year. Works in bull/bear via trend filter; Camarilla levels provide structure; chop filter prevents entries in sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend and chop regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Choppiness Index regime filter (CHOP > 61.8 = ranging, avoid entries)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    chop_period = 14
    atr_12h = np.zeros_like(close_12h)
    for i in range(1, len(close_12h)):
        tr = max(high_12h[i] - low_12h[i], 
                 abs(high_12h[i] - close_12h[i-1]), 
                 abs(low_12h[i] - close_12h[i-1]))
        if i < chop_period:
            atr_12h[i] = np.mean(atr_12h[1:i+1]) if i > 0 else tr
        else:
            atr_12h[i] = (atr_12h[i-1] * (chop_period-1) + tr) / chop_period
    
    # Sum of True Range over chop_period
    tr_sum = np.zeros_like(close_12h)
    for i in range(len(close_12h)):
        if i == 0:
            tr_sum[i] = max(high_12h[i] - low_12h[i], 
                           abs(high_12h[i] - close_12h[i-1]) if i > 0 else 0, 
                           abs(low_12h[i] - close_12h[i-1]) if i > 0 else 0)
        else:
            tr = max(high_12h[i] - low_12h[i], 
                     abs(high_12h[i] - close_12h[i-1]), 
                     abs(low_12h[i] - close_12h[i-1]))
            if i < chop_period:
                tr_sum[i] = np.sum(tr_sum[max(0, i-chop_period+1):i+1]) if i >= chop_period-1 else np.sum(tr_sum[1:i+1]) + tr
            else:
                tr_sum[i] = tr_sum[i-1] - tr_sum[i-chop_period] + tr
    
    # Choppiness Index: 100 * log10(tr_sum / (atr_12h * chop_period)) / log10(chop_period)
    chop_denominator = np.maximum(atr_12h * chop_period, 1e-10)
    chop_ratio = np.divide(tr_sum, chop_denominator, out=np.zeros_like(tr_sum), where=chop_denominator!=0)
    chop_ratio = np.maximum(chop_ratio, 1e-10)
    chop = 100 * np.log10(chop_ratio) / np.log10(chop_period)
    chop[~np.isfinite(chop)] = 50  # set neutral for invalid values
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Get 1d data for Camarilla pivot levels (using 1d for better structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla pivot levels (R3, S3) from previous 1d bar
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 30-period average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (30)
    start_idx = max(50, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Regime filter: only trade when market is NOT choppy (CHOP < 61.8)
        not_choppy = chop_aligned[i] < 61.8
        
        # 12h trend alignment
        trend_12h_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_12h_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0 and not_choppy:
            # Long: price breaks above R3 + volume spike + 12h uptrend
            long_breakout = close[i] > camarilla_r3_aligned[i]
            long_signal = long_breakout and volume_spike[i] and trend_12h_uptrend
            
            # Short: price breaks below S3 + volume spike + 12h downtrend
            short_breakout = close[i] < camarilla_s3_aligned[i]
            short_signal = short_breakout and volume_spike[i] and trend_12h_downtrend
            
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: price touches S3 level OR 12h trend turns down OR chop becomes too high
            if (close[i] < camarilla_s3_aligned[i] or 
                not trend_12h_uptrend or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: price touches R3 level OR 12h trend turns up OR chop becomes too high
            if (close[i] > camarilla_r3_aligned[i] or 
                not trend_12h_downtrend or 
                chop_aligned[i] >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0