#!/usr/bin/env python3
"""
6h_Trix_VolumeSpike_ChopRegime
Hypothesis: Use TRIX (1-period ROC of EMA smoothed) for momentum, volume spikes for confirmation,
and Choppiness Index to filter between trending (TRIX momentum) and ranging (mean reversion) regimes.
In trending markets (CHOP < 38.2): go long when TRIX crosses above zero with volume spike,
short when TRIX crosses below zero with volume spike.
In ranging markets (CHOP > 61.8): fade extreme TRIX values (long when TRIX < -0.1, short when TRIX > 0.1)
with volume spike confirmation. Weekly trend filter (price > weekly EMA50) avoids counter-trend trades.
Designed for 15-30 trades/year to avoid fee drag in 6h timeframe.
"""

name = "6h_Trix_VolumeSpike_ChopRegime"
timeframe = "6h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(14) for Chop
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TRUE RANGE over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # TRIX: 1-period ROC of triple-smoothed EMA (12-period)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix.fillna(0).values
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(close_1w_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(trix[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend (only trade with trend)
        trend_up = close_1w_aligned[i] > ema_50_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Determine regime
            if chop_aligned[i] < 38.2:  # Trending regime
                # Long: TRIX crosses above zero with volume spike
                if trix[i] > 0 and trix[i-1] <= 0 and vol_ratio[i] > 2.0 and trend_up:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX crosses below zero with volume spike
                elif trix[i] < 0 and trix[i-1] >= 0 and vol_ratio[i] > 2.0 and trend_down:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] > 61.8:  # Ranging regime
                # Long: TRIX deeply oversold with volume spike
                if trix[i] < -0.1 and vol_ratio[i] > 2.0:
                    signals[i] = 0.25
                    position = 1
                # Short: TRIX deeply overbought with volume spike
                elif trix[i] > 0.1 and vol_ratio[i] > 2.0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Trending: exit on TRIX cross below zero
                if trix[i] < 0 and trix[i-1] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Ranging: exit when TRIX returns to neutral
                if -0.05 <= trix[i] <= 0.05:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Exit conditions
            if chop_aligned[i] < 38.2:  # Trending: exit on TRIX cross above zero
                if trix[i] > 0 and trix[i-1] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Ranging: exit when TRIX returns to neutral
                if -0.05 <= trix[i] <= 0.05:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals