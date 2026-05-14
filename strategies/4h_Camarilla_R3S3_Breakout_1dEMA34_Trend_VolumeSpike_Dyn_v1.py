#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn_v1
Hypothesis: 4h Camarilla R3/S3 breakouts with 1d EMA34 trend filter and dynamic volume confirmation.
Primary timeframe 4h targets 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.
1d EMA34 provides responsive trend alignment that works in both bull and bear markets.
Dynamic volume confirmation (>1.5x ATR-scaled volume) confirms breakout momentum.
Designed for BTC/ETH with discrete sizing (0.25) to manage drawdown and avoid overtrading.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) on 1d for dynamic volume threshold
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 1d timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate dynamic volume threshold: 1.5x ATR-scaled volume
    vol_atr_ratio = volume / (atr14_1d_aligned * close + 1e-10)  # Avoid division by zero
    vol_threshold = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34, ATR, volume threshold
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Dynamic volume confirmation: current volume > threshold * average volume
            vol_confirm = volume[i] > vol_threshold[i] * pd.Series(volume).rolling(window=20, min_periods=1).mean().iloc[i]
            
            # Long: price breaks above Camarilla R3 in uptrend (price > 1d EMA34) with volume confirmation
            # Short: price breaks below Camarilla S3 in downtrend (price < 1d EMA34) with volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema34_1d_aligned[i]) and vol_confirm
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema34_1d_aligned[i]) and vol_confirm
            
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
            # Exit when price moves back below 1d EMA34 (trend reversal)
            exit_signal = close[i] < ema34_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA34 (trend reversal)
            exit_signal = close[i] > ema34_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_Dyn_v1"
timeframe = "4h"
leverage = 1.0