#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (price > 1d EMA50 for long, < 1d EMA50 for short) and volume confirmation (>2.0x 20-bar mean volume). Uses HTF 1d for trend alignment and HTF 1w for regime filter (only trade when price > 1w EMA200 in bull regime or < 1w EMA200 in bear regime). Designed for low trade frequency (12-37/year) to minimize fee drift while capturing strong breakouts with volume and trend confirmation. Works in bull markets via breakouts above R3 with uptrend filter and in bear markets via breakdowns below S3 with downtrend filter.
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
    
    # Get 1d data for HTF trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 1w data for HTF regime filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(200) on 1w for regime filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 12h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 12h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in direction of 1w trend
        # Bull regime: price > 1w EMA200 → allow longs only
        # Bear regime: price < 1w EMA200 → allow shorts only
        bull_regime = close[i] > ema_200_1w_aligned[i]
        bear_regime = close[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 1d EMA50) with volume confirmation in bull regime
            long_signal = (close[i] > camarilla_r3_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         vol_confirm[i] and \
                         bull_regime
            
            # Short: price breaks below Camarilla S3 in downtrend (price < 1d EMA50) with volume confirmation in bear regime
            short_signal = (close[i] < camarilla_s3_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          vol_confirm[i] and \
                          bear_regime
            
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
            # Exit when price moves back below 1d EMA50 (trend reversal) or breaks below Camarilla S3 (reversal signal)
            exit_signal = (close[i] < ema_50_1d_aligned[i]) or (close[i] < camarilla_s3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal) or breaks above Camarilla R3 (reversal signal)
            exit_signal = (close[i] > ema_50_1d_aligned[i]) or (close[i] > camarilla_r3_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0