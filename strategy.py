#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolFilter
Hypothesis: Camarilla R3/S3 breakouts on 1h with 4h EMA50 trend filter and 1d volume spike confirmation. 
Only trade breakouts in direction of 4h trend when 1d volume > 1.5x 20-period average. 
Uses discrete position sizing (0.20) to minimize fee churn. Target: 20-40 trades/year.
Designed to work in both bull and bear markets via trend alignment and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 4h data (based on previous day's OHLC)
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_R3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_S3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Align HTF indicators to 1h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_R3_4h)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_S3_4h)
    
    # Calculate 1d volume average (20-period) and align
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and Camarilla (1)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or
            np.isnan(camarilla_S3_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Look for Camarilla breakout signals with trend and volume filters
            # Long: price breaks above R3 in uptrend (close > EMA50) with volume spike
            # Short: price breaks below S3 in downtrend (close < EMA50) with volume spike
            volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
            
            long_signal = (close[i] > camarilla_R3_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike
            short_signal = (close[i] < camarilla_S3_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below EMA50 (trend reversal) or Camarilla S3 (mean reversion)
            exit_signal = (close[i] < ema50_aligned[i]) or (close[i] < camarilla_S3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above EMA50 (trend reversal) or Camarilla R3 (mean reversion)
            exit_signal = (close[i] > ema50_aligned[i]) or (close[i] > camarilla_R3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0