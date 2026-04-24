#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with 1w ATR regime filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for ATR-based regime detection (choppy vs trending) and volume spike filter.
- KAMA: Kaufman Adaptive Moving Average adapts to market noise, reducing whipsaws in chop.
- Regime: ATR(10)/ATR(50) ratio > 1.1 = trending (favor KAMA direction), < 0.9 = choppy (avoid trades).
- Entry: Long when price > KAMA AND trending regime AND volume > 1.5 * 20-period average volume.
         Short when price < KAMA AND trending regime AND volume > 1.5 * 20-period average volume.
- Exit: Opposite KAMA cross (price < KAMA for long exit, price > KAMA for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading with KAMA trend in trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w ATR(10) and ATR(50) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for ATR50
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10) and ATR(50)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio for regime: >1.1 = trending, <0.9 = choppy
    atr_ratio = atr10 / atr50
    
    # Align ATR ratio to 1d timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Calculate 1w volume average for confirmation (20-period)
    if len(df_1w) < 20:
        return np.zeros(n)
    
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_len))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) <= er_len else \
                     pd.Series(close).rolling(er_len).apply(lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing Constants
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Calculate KAMA
    kama_vals = kama(close, er_len=10, fast_len=2, slow_len=30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 50)  # Need 30 for ATR10/50 warmup, 50 for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(kama_vals[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade in trending markets (ATR ratio > 1.1)
        trending_regime = atr_ratio_aligned[i] > 1.1
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1w_aligned[i] if not np.isnan(vol_ma_20_1w_aligned[i]) else False
        
        # Exit conditions: opposite KAMA cross
        if position != 0:
            # Exit long: price < KAMA
            if position == 1:
                if curr_close < kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > KAMA
            elif position == -1:
                if curr_close > kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: KAMA cross with regime and volume filters
        if position == 0:
            # Long: price > KAMA AND trending regime AND volume confirmation
            long_condition = (curr_close > kama_vals[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < KAMA AND trending regime AND volume confirmation
            short_condition = (curr_close < kama_vals[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_1wATRRegime_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0