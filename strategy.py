#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1wTrend_Regime
Hypothesis: 4h Camarilla R3/S3 breakout with 1-week EMA trend filter and Bollinger Bandwidth regime filter. Uses HTF 1w EMA for trend alignment (price > 1w EMA for long, < 1w EMA for short) to reduce whipsaw. Bollinger Bandwidth < 0.05 defines low-volatility regime where breakouts are more likely to succeed. Volume confirmation requires >1.8x 20-bar mean volume. Targets 20-30 trades/year per symbol by requiring confluence of trend, regime, and volume. Designed to work in both bull (breakouts with volume in uptrend) and bear (breakouts with volume in downtrend) markets via disciplined entry/exit.
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
    
    # Get 1w data for HTF trend filter (EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align trend EMA to 4h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Bollinger Bandwidth regime filter on 4h (low volatility = BBW < 0.05)
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    low_vol_regime = bb_width < 0.05  # Low volatility regime
    
    # Volume confirmation: current volume > 1.8x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA, BB, and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(bb_mid[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 1w EMA) with low volatility and volume confirmation
            # Short: price breaks below Camarilla S3 in downtrend (price < 1w EMA) with low volatility and volume confirmation
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_1w_aligned[i]) and low_vol_regime[i] and vol_confirm[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_1w_aligned[i]) and low_vol_regime[i] and vol_confirm[i]
            
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
            # Exit when price moves back below Camarilla S3 (mean reversion)
            exit_signal = close[i] < camarilla_s3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R3 (mean reversion)
            exit_signal = close[i] > camarilla_r3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1wTrend_Regime"
timeframe = "4h"
leverage = 1.0