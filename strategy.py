#!/usr/bin/env python3
"""
4h_KAMA_Trend_Filter_With_Volume_Spike
Hypothesis: KAMA identifies trend direction with low lag, combined with volume spike for momentum confirmation.
Go long when KAMA slopes up and volume spikes; short when KAMA slopes down and volume spikes.
Uses 1d EMA as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency (20-50/year) to minimize fee drag while capturing sustained moves.
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
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(change)
    mask = volatility != 0
    er[mask] = change[mask] / volatility[mask]
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # Seed
    for i in range(er_len + 1, n):
        if not np.isnan(sc[i - er_len]):
            kama[i] = kama[i - 1] + sc[i - er_len] * (close[i] - kama[i - 1])
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(er_len + 1, 35)  # Warmup for KAMA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(kama[i-1]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # KAMA slope (direction)
        kama_slope = kama[i] - kama[i-1]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        price = close[i]
        
        if position == 0:
            # Long: KAMA rising + volume spike + price above 1d EMA (uptrend filter)
            if kama_slope > 0 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + volume spike + price below 1d EMA (downtrend filter)
            elif kama_slope < 0 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: KAMA slope turns down OR price breaks below 1d EMA
            if kama_slope <= 0 or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: KAMA slope turns up OR price breaks above 1d EMA
            if kama_slope >= 0 or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_Filter_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0