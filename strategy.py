#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume confirmation.
Primary timeframe 1h targets 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
4h EMA50 provides responsive trend alignment that works in both bull and bear markets.
Volume confirmation (>1.5x ATR-scaled volume) ensures breakout momentum.
Designed for BTC/ETH with discrete sizing (0.20) to manage drawdown and avoid overtrading.
Session filter (08-20 UTC) reduces noise trades.
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
    
    # Get 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA50 on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate ATR(14) on 4h for dynamic volume threshold
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr14_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior bar)
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h)  # R1 = C + 1.1*(H-L)
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h)  # S1 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate dynamic volume threshold: 1.5x ATR-scaled volume
    vol_atr_ratio = volume / (atr14_4h_aligned * close + 1e-10)  # Avoid division by zero
    vol_threshold = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, ATR, volume threshold
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Dynamic volume confirmation: current volume > threshold * average volume
            vol_confirm = volume[i] > vol_threshold[i] * pd.Series(volume).rolling(window=20, min_periods=1).mean().iloc[i]
            
            # Long: price breaks above Camarilla R1 in uptrend (price > 4h EMA50) with volume confirmation
            # Short: price breaks below Camarilla S1 in downtrend (price < 4h EMA50) with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_4h_aligned[i]) and vol_confirm
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_4h_aligned[i]) and vol_confirm
            
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
            # Exit when price moves back below 4h EMA50 (trend reversal)
            exit_signal = close[i] < ema50_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 4h EMA50 (trend reversal)
            exit_signal = close[i] > ema50_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0