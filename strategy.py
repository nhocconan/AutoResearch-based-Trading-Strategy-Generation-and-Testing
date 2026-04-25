#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1wTrend_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts with 1-week EMA50 trend filter, 1d EMA34 secondary trend confirmation, and volume spike (>2.0x 20-bar avg) produce high-quality entries. Uses discrete sizing (0.25) to limit trades (~20-40/year) and avoid fee drag. The weekly EMA50 provides strong trend alignment for multi-month moves, while 1d EMA34 adds intermediate-term filter. Volume confirms breakout momentum. Designed for BTC/ETH robustness across bull/bear regimes via strict multi-timeframe trend alignment and volume confirmation, minimizing whipsaws in ranging markets.
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
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior day)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1w data for stronger trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for stronger trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-bar average volume for confirmation on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34, EMA50, volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average (strict filter)
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R3 with both weekly and daily uptrend and volume spike
            # Short: price breaks below Camarilla S3 with both weekly and daily downtrend and volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and \
                         (close[i] > ema50_1w_aligned[i]) and \
                         (close[i] > ema34_1d_aligned[i]) and \
                         volume_confirm
            short_signal = (close[i] < camarilla_s3_aligned[i]) and \
                          (close[i] < ema50_1w_aligned[i]) and \
                          (close[i] < ema34_1d_aligned[i]) and \
                          volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below either weekly or daily EMA (trend weakening)
            exit_signal = (close[i] < ema50_1w_aligned[i]) or (close[i] < ema34_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above either weekly or daily EMA (trend weakening)
            exit_signal = (close[i] > ema50_1w_aligned[i]) or (close[i] > ema34_1d_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1wTrend_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0