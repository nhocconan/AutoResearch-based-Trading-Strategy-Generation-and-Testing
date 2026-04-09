#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w ADX trend filter + 1d volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# 1w ADX > 25 filters for strong trending markets (avoids chop)
# 1d volume > 1.5 * 20-period average volume confirms breakout strength
# Works in bull/bear: ADX ensures we only trade strong trends, Elder Ray captures momentum
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1w_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w < 50):
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Elder Ray and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # +DM and -DM
    up_move = high_1w[1:] - high_1w[:-1]
    down_move = low_1w[:-1] - low_1w[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Wilder's smoothing for TR, +DM, -DM
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di_1w = np.where(atr_1w > 0, 100 * plus_dm_smooth / atr_1w, 0.0)
    minus_di_1w = np.where(atr_1w > 0, 100 * minus_dm_smooth / atr_1w, 0.0)
    
    # DX and ADX
    dx_1w = np.where((plus_di_1w + minus_di_1w) > 0, 
                     100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w), 
                     0.0)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 6h timeframe (wait for 1w bar close)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d Elder Ray to 6h timeframe (wait for 1d bar close)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d volume confirmation: volume > 1.5 * 20-period average volume
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_1d = volume_1d > 1.5 * avg_volume_1d
    volume_confirmed_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmed_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1w_aligned[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_confirmed_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w ADX > 25 indicates strong trend
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes negative OR trend weakens
            if bear_power_1d_aligned[i] < 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes positive OR trend weakens
            if bull_power_1d_aligned[i] > 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: only in strong trend
            if strong_trend:
                # Long when Bull Power > 0 and volume confirmed
                if bull_power_1d_aligned[i] > 0 and volume_confirmed_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short when Bear Power < 0 and volume confirmed
                elif bear_power_1d_aligned[i] < 0 and volume_confirmed_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals