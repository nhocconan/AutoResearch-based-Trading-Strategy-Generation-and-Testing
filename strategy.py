#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2
Hypothesis: Tighten entry conditions from v1 by requiring volume > 2.5x 20-bar mean (vs 2.0x) and adding ATR-based volatility filter (ATR(14) > 0.5 * ATR(50)) to ensure breakouts occur in sufficient volatility regimes. This reduces trade frequency to target 15-25 trades/year while maintaining edge in both bull (breakouts with volume/vol) and bear (trend-following via shorts) markets. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of prior bar)
    camarilla_r3 = close_12h + 1.1 * (high_12h - low_12h)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_12h - 1.1 * (high_12h - low_12h)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: current volume > 2.5x 20-bar mean volume (tighter than v1)
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.5)
    
    # ATR-based volatility filter: ATR(14) > 0.5 * ATR(50) ensures sufficient volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA, volume mean, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_mean_20[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 12h EMA50) with volume confirmation and volatility filter
            # Short: price breaks below Camarilla S3 in downtrend (price < 12h EMA50) with volume confirmation and volatility filter
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i] and vol_filter[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i] and vol_filter[i]
            
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
            # Exit when price moves back below 12h EMA50 (trend reversal)
            exit_signal = close[i] < ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 12h EMA50 (trend reversal)
            exit_signal = close[i] > ema_50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0