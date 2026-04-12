#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_trix_volume_chop_v2
# Uses TRIX (15-period triple EMA) for momentum, volume spike (2x 20-period avg),
# and Choppiness Index (14-period) to filter regimes: CHOP > 61.8 for mean-reversion mode.
# Long when TRIX crosses above zero with volume confirmation in choppy market.
# Short when TRIX crosses below zero with volume confirmation in choppy market.
# Exit when TRIX crosses back across zero.
# Designed for low trade frequency (target: 15-40 trades/year) to minimize fee drag.
# Works in ranging markets via mean-reversion momentum and avoids trending whipsaw via chop filter.

name = "4h_1d_trix_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily True Range and ADX-like components for Choppiness
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])  # seed with simple average
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period_chop = 14
    atr = wilders_smoothing(tr, period_chop)
    plus_di = 100 * wilders_smoothing(plus_dm, period_chop) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, period_chop) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, period_chop)
    
    # Choppiness Index: higher = more choppy, lower = more trending
    chop = 100 * np.log10(atr.sum() / (np.abs(plus_di - minus_di).sum())) / np.log10(period_chop)
    # Fix: compute rolling sum correctly
    tr_sum = pd.Series(tr).rolling(window=period_chop, min_periods=period_chop).sum().values
    dm_sum = np.abs(pd.Series(plus_di - minus_di).rolling(window=period_chop, min_periods=period_chop).sum().values)
    chop = 100 * np.log10(tr_sum / dm_sum) / np.log10(period_chop)
    
    # Align Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX (15-period triple EMA of ROC)
    # ROC = (close - close.prev) / close.prev * 100
    roc = np.diff(close) / close[:-1] * 100
    roc = np.concatenate([[np.nan], roc])  # align length
    
    # Triple EMA
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # already in %
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade in choppy/range-bound markets (CHOP > 61.8)
        if chop_aligned[i] <= 61.8:
            # In trending markets, stay flat to avoid whipsaw
            signals[i] = 0.0
            position = 0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # TRIX crossover signals
        trix_prev = trix[i-1] if i > 0 else 0
        
        # Long when TRIX crosses above zero
        if trix[i] > 0 and trix_prev <= 0 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short when TRIX crosses below zero
        elif trix[i] < 0 and trix_prev >= 0 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit when TRIX crosses back across zero
        elif position == 1 and trix[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and trix[i] > 0:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals