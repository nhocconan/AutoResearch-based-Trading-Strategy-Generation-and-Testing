# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# 1D_KAMA_RIBBON_VOLUME_V1
# Hypothesis: On daily timeframe, KAMA ribbon (fast/slow) crossover signals trend changes,
# filtered by volume spike (2x average) and weekly ADX trend strength (>25).
# Uses weekly trend filter to avoid counter-trend trades in choppy markets.
# Designed for low frequency (target 15-25 trades/year) to minimize fee drag in 2025 bear market.
# Works in both bull and bear: KAMA adapts to market noise, volume confirms institutional interest.

name = "1D_KAMA_RIBBON_VOLUME_V1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for ADX trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- KAMA Ribbon (fast: 2, slow: 30) ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:10])  # placeholder for rolling sum
    # Proper ER calculation
    price_change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(close - np.roll(close, 10))
    er_den = np.zeros(n)
    for i in range(10, n):
        er_den[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(er_den > 0, er_num / er_den, 0)
    
    # Smoothing constants
    fast_sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    slow_sc = (er * (2/(5+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama_fast = np.zeros(n)
    kama_slow = np.zeros(n)
    kama_fast[0] = close[0]
    kama_slow[0] = close[0]
    for i in range(1, n):
        kama_fast[i] = kama_fast[i-1] + fast_sc[i] * (close[i] - kama_fast[i-1])
        kama_slow[i] = kama_slow[i-1] + slow_sc[i] * (close[i] - kama_slow[i-1])
    
    # --- Weekly ADX (14-period) for trend strength ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - np.roll(close_1w, 1)[1:])
    tr3 = np.abs(low_1w[1:] - np.roll(close_1w, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # --- Volume Spike (daily) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    # Align weekly ADX to daily
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if ADX is not ready
        if np.isnan(adx_aligned[i]):
            if position != 0:
                # Exit if KAMA ribbon crosses opposite direction
                if position == 1 and kama_fast[i] < kama_slow[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and kama_fast[i] > kama_slow[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Entry conditions: KAMA ribbon cross + volume spike + strong trend (ADX > 25)
        kama_cross_up = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_down = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        long_entry = kama_cross_up and vol_spike[i] and (adx_aligned[i] > 25)
        short_entry = kama_cross_down and vol_spike[i] and (adx_aligned[i] > 25)
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        else:
            # Exit on KAMA ribbon reverse or weak trend
            if position == 1:
                if (kama_fast[i] < kama_slow[i]) or (adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if (kama_fast[i] > kama_slow[i]) or (adx_aligned[i] < 20):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals