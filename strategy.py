#!/usr/bin/env python3
"""
6h_Adaptive_Kelly_TRIX_Trend_Filter
Hypothesis: Combines TRIX momentum with volatility-adjusted Kelly sizing for 6h timeframe.
Uses TRIX(12) crossing zero as momentum signal, filtered by 1d EMA50 trend, with position
size scaled by inverse volatility (ATR-based) and Kelly criterion based on recent win rate.
Designed for low trade frequency (<30/year) with strong edge in both bull and bear markets
by adapting position size to volatility and maintaining trend alignment.
Timeframe: 6h
"""

name = "6h_Adaptive_Kelly_TRIX_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for EMA50 trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate ATR(14) for volatility normalization
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first value
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values

    # Calculate TRIX(12,9,9) - triple EMA of ROC
    # ROC = (close / close.shift(1) - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    
    # Triple EMA
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3

    # Calculate recent win rate for Kelly (lookback 50 periods)
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] / close[:-1] - 1)
    
    win_rate = np.full_like(close, 0.5)  # default 50%
    for i in range(50, n):
        # Look at returns when we had signals in past 50 periods
        # Simplified: use price change direction as proxy
        period_returns = returns[i-50:i]
        wins = np.sum(period_returns > 0)
        win_rate[i] = wins / 50.0 if 50 > 0 else 0.5
    
    # Kelly fraction: f = (bp * w - l) / b where b=1 (1:1), w=win_rate, l=loss_rate
    kelly = np.maximum(0, win_rate * 2 - 1)  # simplified for 1:1 payoff
    kelly = np.minimum(kelly, 0.5)  # cap at 50% kelly
    
    # Volatility scaling: inverse of ATR normalized
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=20).mean().values
    vol_scale = np.ones_like(atr)
    mask = atr_ma > 0
    vol_scale[mask] = np.clip(atr[mask] / atr_ma[mask], 0.5, 2.0)  # normalize
    vol_scale = 1.0 / vol_scale  # inverse - lower vol = higher scale
    vol_scale = np.clip(vol_scale, 0.5, 2.0)  # cap scaling

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):  # Start after warmup
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_scale[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # TRIX signal: crossing zero
        trix_signal = 0
        if i > 0:
            if trix[i-1] <= 0 and trix[i] > 0:
                trix_signal = 1   # bullish crossover
            elif trix[i-1] >= 0 and trix[i] < 0:
                trix_signal = -1  # bearish crossover

        if position == 0 and trix_signal != 0:
            # Check trend filter: price vs 1d EMA50
            if trix_signal == 1 and close[i] > ema_50_1d_aligned[i]:
                # Long signal
                base_size = 0.25
                kelly_adj = kelly[i] * 0.5  # use half Kelly
                vol_adj = vol_scale[i]
                size = base_size * kelly_adj * vol_adj
                size = np.clip(size, 0.1, 0.35)  # reasonable bounds
                signals[i] = size
                position = 1
            elif trix_signal == -1 and close[i] < ema_50_1d_aligned[i]:
                # Short signal
                base_size = 0.25
                kelly_adj = kelly[i] * 0.5  # use half Kelly
                vol_adj = vol_scale[i]
                size = base_size * kelly_adj * vol_adj
                size = np.clip(size, 0.1, 0.35)  # reasonable bounds
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: TRIX turns negative or trend breaks
            if trix[i] < 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Exit short: TRIX turns positive or trend breaks
            if trix[i] > 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position

    return signals