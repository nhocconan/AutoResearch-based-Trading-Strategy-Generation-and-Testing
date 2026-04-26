#!/usr/bin/env python3
"""
6h_KAMA_AdaptiveTrend_1dRegime_Filter
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise - fast in trends, slow in chop.
Combined with 1d regime filter (ADX + chop) to avoid whipsaws. Long when price > KAMA and bullish regime,
short when price < KAMA and bearish regime. Uses 6h for institutional trend following with adaptive smoothing.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 6h data
    # Efficiency Ratio (ER) = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(np.subtract(close, np.roll(close, 10)))  # 10-period net change
    er_den = np.sum(np.lib.stride_tricks.sliding_window_view(abs_change, 10), axis=1)
    # Pad beginning for rolling sum
    er_den_padded = np.concatenate([np.full(9, np.nan), er_den])
    er = np.where(er_den_padded != 0, er_num / er_den_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1d ADX for trend strength
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - np.roll(close_1d, 1)[1:])
    tr3 = np.abs(low_1d[1:] - np.roll(close_1d, 1)[1:])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1 / period
    atr = np.full_like(tr, np.nan)
    atr[period] = np.nansum(tr[1:period+1])  # initial seed
    for i in range(period+1, len(tr)):
        atr[i] = atr[i-1] * (1 - alpha) + alpha * tr[i]
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=alpha, adjust=False).mean().values
    
    # Calculate 1d Choppiness Index
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    high_low_range = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values - \
                     pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / high_low_range) / np.log10(chop_period)
    
    # Align HTF indicators
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is already 6h but we align for consistency
    
    # Regime filters
    strong_trend = adx_aligned > 25
    chopping_market = chop_aligned > 61.8
    trending_market = chop_aligned < 38.2
    
    # Volume confirmation on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, 20)  # ADX period and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(adx_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Price relative to KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Long: Price > KAMA, strong trend OR trending market (not choppy), volume confirmation
            if price_above_kama and (strong_trend[i] or trending_market[i]) and not chopping_market[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA, strong trend OR trending market (not choppy), volume confirmation
            elif price_below_kama and (strong_trend[i] or trending_market[i]) and not chopping_market[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price crosses below KAMA OR market becomes choppy
            if price_below_kama or chopping_market[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price crosses above KAMA OR market becomes choppy
            if price_above_kama or chopping_market[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_KAMA_AdaptiveTrend_1dRegime_Filter"
timeframe = "6h"
leverage = 1.0