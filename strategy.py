#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ChopFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter (price > 1d EMA50 for long, < 1d EMA50 for short), volume confirmation (>2.0x 20-bar mean volume), and choppiness regime filter (CHOP(14) < 61.8 for trending markets). Uses HTF 1d for trend alignment and regime detection to capture medium-term momentum while reducing whipsaw in choppy markets. Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn. Designed for 20-40 trades/year per symbol, effective in both bull (breakouts with volume) and bear (trend-following via shorts) markets by avoiding false signals in ranging conditions.
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
    
    # Get 1d data for HTF trend filter and choppiness regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate Choppiness Index on 1d: CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (HHV(14,high) - LLV(14,low)))
    # Simplified: CHOP(14) = 100 * log10( sum(tr) / log10(14) / (max(high) - min(low)) ) over 14 periods
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])  # align with index
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high_14 - lowest_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, sum_atr_14 / range_14, np.nan)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    chop = np.where(np.isnan(chop), 50.0, chop)  # neutral when undefined
    
    # Align HTF indicators to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)  # R3 = C + 1.1*(H-L)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)  # S3 = C - 1.1*(H-L)
    
    # Align Camarilla levels to 4h timeframe (use previous bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0x 20-bar mean volume
    vol_mean_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_mean_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA, CHOP, and volume mean
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_mean_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        in_trending_regime = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend (price > 1d EMA50) with volume confirmation and trending regime
            # Short: price breaks below Camarilla S3 in downtrend (price < 1d EMA50) with volume confirmation and trending regime
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema_50_aligned[i]) and vol_confirm[i] and in_trending_regime
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema_50_aligned[i]) and vol_confirm[i] and in_trending_regime
            
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
            # Exit when price moves back below 1d EMA50 (trend reversal) OR enters choppy regime
            exit_signal = (close[i] < ema_50_aligned[i]) or (chop_aligned[i] >= 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal) OR enters choppy regime
            exit_signal = (close[i] > ema_50_aligned[i]) or (chop_aligned[i] >= 61.8)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0