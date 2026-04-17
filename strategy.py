#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-day volume spike and ADX trend filter
# Camarilla levels from daily pivots provide institutional support/resistance.
# Breakouts with volume confirmation and ADX>25 filter capture true trends.
# Designed for 12h timeframe to achieve 12-37 trades/year with low fee decay.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1-day Camarilla Pivot Levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla R1 and S1
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1-day Volume Spike (vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1-day ADX (14-period) for trend filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 12-hour Close for Breakout Detection ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume spike: current 1d volume > 1.5x 20-period average
        vol_spike = volume_1d[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Trend filter: ADX > 25
        trend_filter = adx_aligned[i] > 25
        
        # Camarilla breakout signals
        breakout_up = close_12h_aligned[i] > r1_aligned[i]
        breakout_down = close_12h_aligned[i] < s1_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            if vol_spike and trend_filter:
                if breakout_up:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic: exit when price returns to pivot or conditions fail
        elif position == 1:
            # Exit long if price returns to pivot or conditions fail
            if close_12h_aligned[i] < pivot[i] or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price returns to pivot or conditions fail
            if close_12h_aligned[i] > pivot[i] or not vol_spike or not trend_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ADXFilter"
timeframe = "12h"
leverage = 1.0