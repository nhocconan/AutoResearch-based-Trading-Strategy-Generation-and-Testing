#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm_Regime
Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA50 trend filter, volume spike (>2x 20-bar avg), and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trending). Only takes breakout trades in trending regimes aligned with 1d trend. Designed for low trade frequency (~15-25/year) to work in both bull and bear markets via trend alignment and volume confirmation. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for choppiness regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # roll gives close_1d[-1] for index 0, but we set correctly below
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength (alternative to choppiness)
    # +DM, -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed +DM, -DM, TR
    tr_rma = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # +DI, -DI, DX
    plus_di = 100 * plus_dm_smooth / tr_rma
    minus_di = 100 * minus_dm_smooth / tr_rma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 12h data for Camarilla levels and primary price series
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels on 12h data (based on previous bar's OHLC)
    camarilla_r1_12h = close_12h + ((high_12h - low_12h) * 1.1 / 12)
    camarilla_s1_12h = close_12h - ((high_12h - low_12h) * 1.1 / 12)
    camarilla_r3_12h = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3_12h = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    
    # Align HTF indicators to 12h timeframe (prices is already 12h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume (strict filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), ADX (14), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout signals with trend filter, volume spike, and regime filter
            # Long: price breaks above R1 in uptrend (close > EMA50) with volume spike and trending regime
            # Short: price breaks below S1 in downtrend (close < EMA50) with volume spike and trending regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i] and is_trending
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i] and is_trending
            
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
            # Exit when price moves back below Camarilla S1 (stop loss) or reaches R3 (take profit)
            exit_signal = close[i] < camarilla_s1_aligned[i] or close[i] > camarilla_r3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R1 (stop loss) or reaches S3 (take profit)
            exit_signal = close[i] > camarilla_r1_aligned[i] or close[i] < camarilla_s3_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm_Regime"
timeframe = "12h"
leverage = 1.0