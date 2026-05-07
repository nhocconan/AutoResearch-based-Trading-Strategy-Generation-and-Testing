#!/usr/bin/env python3
"""
4H_Trix_Volume_Spike_Regime_v1
Hypothesis: TRIX (15) crossing zero with volume spike (>2x 20-period avg) in trending regime (ADX > 25) captures momentum bursts.
Long when TRIX crosses above zero; short when crosses below zero. Volume confirms conviction, ADX filter avoids chop.
Works in bull/bear by catching strong moves; avoids whipsaws via regime filter. Target: 20-40 trades/year.
"""
name = "4H_Trix_Volume_Spike_Regime_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 4h data for TRIX and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15,9,9) - triple EMA of 1-period % change
    close_4h = pd.Series(df_4h['close'])
    roc = close_4h.pct_change(1)
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3 * 100  # scale for readability
    trix_prev = trix.shift(1)
    trix = trix.fillna(0).values
    trix_prev = trix_prev.fillna(0).values
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    trix_prev_aligned = align_htf_to_ltf(prices, df_4h, trix_prev)
    
    # Calculate ADX (14) for regime filter
    plus_dm = pd.Series(np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0))
    minus_dm = pd.Series(np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0))
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = pd.Series(np.maximum(np.maximum(tr1, tr2), tr3))
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx = np.concatenate([np.full(14, np.nan), adx.values])
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume filter: current volume > 2.0 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_prev_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.3 days on 4h TF)
            if bars_since_exit < 8:
                continue
                
            # Long: TRIX crosses above zero + volume spike + ADX > 25 (trending)
            if (trix[i] > 0 and trix_prev[i] <= 0 and 
                volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: TRIX crosses below zero + volume spike + ADX > 25
            elif (trix[i] < 0 and trix_prev[i] >= 0 and 
                  volume_spike[i] and adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: TRIX crosses back through zero
            if position == 1 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals