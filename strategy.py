#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Chop Regime Filter
# - TRIX (15-period) measures momentum with reduced noise
# - Long when TRIX crosses above zero with volume > 2x 20-period average and chop > 61.8 (range)
# - Short when TRIX crosses below zero with volume > 2x 20-period average and chop > 61.8 (range)
# - Uses chop regime to avoid whipsaws in strong trends (chop < 38.2) and focus on mean reversion in ranges
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits for 4h
# - Works in both bull (mean reversion in rallies) and bear (mean reversion in declines) markets

name = "4h_trix_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return signals
    
    # Pre-compute 1d Chop Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_1d = []
    for i in range(len(close_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    
    atr_1d = np.array(atr_1d)
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    high_max_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_min_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(atr_sum_14 / np.log10(14) / (high_max_14 - low_min_14))
    chop_1d = np.where((high_max_14 - low_min_14) == 0, 50, chop_1d)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Pre-compute TRIX (15-period) on 4h data
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value has no previous
    
    # Pre-compute 4h volume SMA (20-period)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(15, n):  # Start after warmup for TRIX and chop
        # Skip if any required data is invalid
        if (np.isnan(trix[i]) or np.isnan(chop_1d_aligned[i]) or np.isnan(volume_sma_20[i]) or
            np.isnan(trix[i-1]) if i > 0 else False):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        volume_current = volume[i]
        
        # TRIX crossover signals
        trix_cross_above = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_below = trix[i] < 0 and trix[i-1] >= 0
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume_current > 2.0 * volume_sma_20[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (good for mean reversion)
        chop_range = chop_1d_aligned[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: TRIX crosses above zero + volume confirmation + chop range
        if trix_cross_above and vol_confirm and chop_range:
            enter_long = True
        
        # Short: TRIX crosses below zero + volume confirmation + chop range
        if trix_cross_below and vol_confirm and chop_range:
            enter_short = True
        
        # Exit conditions: opposite TRIX crossover or chop leaves range (trend developing)
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if TRIX crosses below zero OR chop drops below 50 (trend developing)
            exit_long = trix_cross_below or (chop_1d_aligned[i] < 50)
        elif position == -1:
            # Exit short if TRIX crosses above zero OR chop drops below 50 (trend developing)
            exit_short = trix_cross_above or (chop_1d_aligned[i] < 50)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals