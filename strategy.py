#!/usr/bin/env python3
"""
Hypothesis: 12h TRIX with volume spike and 1d choppiness regime filter.
- Primary timeframe: 12h for execution, HTF: 1d for chop regime and volume confirmation.
- TRIX: Triple EMA of closing price, momentum oscillator. Long when TRIX crosses above zero,
  short when crosses below zero. Uses 12-period EMA (standard) to reduce lag.
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Choppiness regime: only trade when 1d CHOP(14) > 61.8 (range market) for mean reversion,
  or when CHOP(14) < 38.2 (trending market) for trend following. Adaptive logic.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via trend following when chop low, in bear via mean reversion when chop high.
- Uses EMA for smoothness and reduced whipsaw vs SMA.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Exponential Moving Average with min_periods"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def trix(close, period=12):
    """TRIX: Triple EMA of closing price, then percent change"""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    # Calculate percent change: (current - previous) / previous * 100
    trix_val = np.full_like(e3, np.nan, dtype=float)
    trix_val[1:] = (e3[1:] - e3[:-1]) / e3[:-1] * 100
    return trix_val

def choppiness_index(high, low, close, period=14):
    """Choppiness Index: measures whether market is choppy (ranging) or trending"""
    if len(high) < period:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.max(tr1[0], tr2[0], tr3[0]) if len(tr1) > 0 else 0], tr])  # first TR
    
    # Sum of TR over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    max_hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Chop formula: 100 * log10(tr_sum / (max_hh - min_ll)) / log10(period)
    chop = np.full_like(high, np.nan, dtype=float)
    denominator = max_hh - min_ll
    valid = (denominator > 0) & ~np.isnan(tr_sum) & ~np.isnan(denominator)
    chop[valid] = 100 * np.log10(tr_sum[valid] / denominator[valid]) / np.log10(period)
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX on 1d close
    trix_1d = trix(close_1d, 12)
    trix_1d_aligned = align_htf_to_ltf(prices, df_1d, trix_1d)
    
    # Choppiness Index on 1d
    chop_1d = choppiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 14)  # volume MA + TRIX + chop
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Adaptive logic based on chop regime
            if chop_1d_aligned[i] < 38.2:  # Trending market - follow TRIX
                if i > 0 and not np.isnan(trix_1d_aligned[i-1]):
                    # TRIX crossing above zero = long signal
                    if trix_1d_aligned[i-1] <= 0 and trix_1d_aligned[i] > 0:
                        if volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                    # TRIX crossing below zero = short signal
                    elif trix_1d_aligned[i-1] >= 0 and trix_1d_aligned[i] < 0:
                        if volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
            elif chop_1d_aligned[i] > 61.8:  # Ranging market - mean reversion at extremes
                # In ranging markets, look for TRIX extremes for mean reversion
                if i > 0 and not np.isnan(trix_1d_aligned[i-1]):
                    # TRIX very negative and turning up = long (oversold bounce)
                    if (trix_1d_aligned[i-1] < -0.5 and trix_1d_aligned[i] > trix_1d_aligned[i-1] and
                        trix_1d_aligned[i] < 0):
                        if volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                    # TRIX very positive and turning down = short (overbought pullback)
                    elif (trix_1d_aligned[i-1] > 0.5 and trix_1d_aligned[i] < trix_1d_aligned[i-1] and
                          trix_1d_aligned[i] > 0):
                        if volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero (trending) or TRIX declines from positive (ranging)
            if chop_1d_aligned[i] < 38.2:  # Trending - exit on TRIX cross down
                if trix_1d_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging - exit when TRIX returns to zero or declines
                if trix_1d_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero (trending) or TRIX rises from negative (ranging)
            if chop_1d_aligned[i] < 38.2:  # Trending - exit on TRIX cross up
                if trix_1d_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging - exit when TRIX returns to zero or rises
                if trix_1d_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0