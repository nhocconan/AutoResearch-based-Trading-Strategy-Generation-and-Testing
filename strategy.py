#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_RegimeADX
Hypothesis: Camarilla R3/S3 breakouts on 6h with 1d EMA50 trend filter, ADX regime filter (ADX > 25 = trending), and volume confirmation (>1.5x 20-bar average). 
Breakouts at R3/S3 (strong intraday levels) with trend alignment reduce false signals. ADX filter ensures we only trade in trending markets where breakouts work.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year (50-150 over 4 years).
Designed for 6h timeframe to work in both bull and bear markets by combining intraday structure with daily trend and regime filters.
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
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ADX on 1d for regime filter (trending when ADX > 25)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = high_1d[0] - close_1d[0]
    tr3[0] = low_1d[0] - close_1d[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and TR
    tr_period = 14
    atr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    minus_di = 100 * minus_dm_smooth / np.where(atr_smooth == 0, 1, atr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx_1d = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate Camarilla levels on 1d data (based on previous bar's OHLC)
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    camarilla_s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Align HTF indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d, additional_delay_bars=1)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume (moderate filter to balance trade frequency)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50), ADX (~28), and volume MA (20)
    start_idx = max(55, 20)  # EMA50 needs 50, others need less
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 1d trend with ADX regime filter and volume confirmation
            # Long: price breaks above R3 in uptrend (close > EMA50) with ADX > 25 (trending) and volume spike
            # Short: price breaks below S3 in downtrend (close < EMA50) with ADX > 25 (trending) and volume spike
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_aligned[i]) and (adx_aligned[i] > 25) and volume_spike[i]
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_aligned[i]) and (adx_aligned[i] > 25) and volume_spike[i]
            
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
            # Exit when price moves back below Camarilla R4 (take profit at stronger resistance) or trend changes
            exit_signal = (close[i] < camarilla_r4_aligned[i]) or (close[i] < ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla S4 (take profit at stronger support) or trend changes
            exit_signal = (close[i] > camarilla_s4_aligned[i]) or (close[i] > ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_RegimeADX"
timeframe = "6h"
leverage = 1.0