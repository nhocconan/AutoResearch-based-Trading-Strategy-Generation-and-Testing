#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Regime_ADX
Hypothesis: Camarilla R3/S3 breakout with 1d EMA50 trend filter, ADX regime filter (ADX>25 for trending), and volume confirmation. Designed for low trade frequency (<50/year) and robustness in both bull and bear markets.
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
    
    # Get 4h data for Camarilla calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for each 4h bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (Range * 1.1 / 2)
    # S3 = C - (Range * 1.1 / 2)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = close_4h + (range_4h * 1.1 / 2.0)
    s3_4h = close_4h - (range_4h * 1.1 / 2.0)
    
    # Align Camarilla levels to 15m timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Get 1d data for trend filter (EMA50) and ADX regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ADX for regime filter (trending when ADX > 25)
    # ADX calculation: +DI, -DI, DX, then ADX = smoothed DX
    period = 14
    up_move = np.diff(high_1d)
    down_move = -np.diff(low_1d)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # True Range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(np.diff(low_1d))
    tr3 = np.abs(np.diff(close_1d))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1d = np.zeros_like(tr)
    if len(tr) > 0:
        atr_1d[0] = tr[0]
        for i in range(1, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * (period-1) + tr[i]) / period
    
    # Avoid division by zero
    atr_1d_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr_1d_safe
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr_1d_safe
    
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) == 0, 1e-10, (plus_di_1d + minus_di_1d))
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        if position == 0:
            # Only trade in trending regime (ADX > 25)
            if adx_val > 25.0:
                if close[i] > ema_trend:  # Uptrend regime
                    # Long: break above R3 with volume spike
                    long_signal = (close[i] > r3_aligned[i]) and vol_spike[i]
                    # Short: break below S3 only if extreme volume spike (counter-trend fade)
                    short_signal = (close[i] < s3_aligned[i]) and vol_spike[i] and (volume[i] > (3.0 * vol_ma_20[i]))
                else:  # Downtrend regime
                    # Short: break below S3 with volume spike
                    short_signal = (close[i] < s3_aligned[i]) and vol_spike[i]
                    # Long: break above R3 only if extreme volume spike (counter-trend fade)
                    long_signal = (close[i] > r3_aligned[i]) and vol_spike[i] and (volume[i] > (3.0 * vol_ma_20[i]))
            else:
                long_signal = False
                short_signal = False
            
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
            # Exit conditions: re-enter S3 or trend reversal (ADX drops)
            exit_signal = (close[i] < s3_aligned[i]) or (adx_val < 20.0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: re-enter R3 or trend reversal (ADX drops)
            exit_signal = (close[i] > r3_aligned[i]) or (adx_val < 20.0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Regime_ADX"
timeframe = "4h"
leverage = 1.0