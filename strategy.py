#!/usr/bin/env python3
# 1D_TRIX_VolumeSpike_Trend
# Hypothesis: TRIX momentum with volume spike confirmation and weekly trend filter.
# Long when TRIX crosses above zero with volume > 2x average and weekly uptrend.
# Short when TRIX crosses below zero with volume > 2x average and weekly downtrend.
# Uses volatility filter to avoid choppy markets.
# Target: 15-25 trades/year per symbol.

name = "1D_TRIX_VolumeSpike_Trend"
timeframe = "1d"
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
    
    # TRIX calculation (15-period)
    close_s = pd.Series(close)
    # Triple EMA
    ema1 = close_s.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    # Percent change
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # Volume average (30-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=30, min_periods=30).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volatility filter: avoid high chop (ATR ratio)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    atr = np.concatenate([np.full(1, np.nan), atr])
    # ATR ratio: current ATR / 50-period average ATR
    atr_s = pd.Series(atr)
    atr_ma = atr_s.rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma
    # Only trade when ATR ratio < 1.5 (not too volatile)
    low_volatility = atr_ratio < 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(vol_ma[i]) or np.isnan(weekly_uptrend_aligned[i]) or
            np.isnan(weekly_downtrend_aligned[i]) or np.isnan(low_volatility[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        low_vol = low_volatility[i]
        
        if position == 0:
            # Enter long: TRIX bullish crossover + volume spike + weekly uptrend + low volatility
            if i > 0 and trix[i-1] <= 0 and trix[i] > 0 and volume_spike and weekly_up and low_vol:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX bearish crossover + volume spike + weekly downtrend + low volatility
            elif i > 0 and trix[i-1] >= 0 and trix[i] < 0 and volume_spike and weekly_down and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX turns bearish or volatility spikes
            if trix[i] < 0 or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX turns bullish or volatility spikes
            if trix[i] > 0 or not low_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals