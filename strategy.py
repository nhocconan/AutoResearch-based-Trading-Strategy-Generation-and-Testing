#!/usr/bin/env python3
"""
6h_KAMA_Regime_Adaptive_Consensus
Hypothesis: Adaptive strategy that switches between trend-following and mean-reversion based on 1d market regime (ADX). Uses 6h Kaufman Adaptive Moving Average (KAMA) as primary signal generator with volume confirmation. In trending regimes (ADX>25): go long when price > KAMA + 0.5*ATR, short when price < KAMA - 0.5*ATR. In ranging regimes (ADX<20): fade extremes using Bollinger Bands (20,2) on 6h - long at lower band, short at upper band. Volume confirmation (>1.5x 20-period avg) required for all entries. Designed for low trade frequency (<30/year) via regime filtering and strict entry conditions.
"""

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
    
    # Get 1d data for regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for ADX
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ADX(14) for regime detection
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h KAMA(10,2,30) - ER=10, fastest=2, slowest=30
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else \
                 np.array([np.sum(np.abs(np.diff(close[i-10:i+1]))) if i>=10 else 0 for i in range(len(close))])
    # Simplified volatility calculation for 10-period
    volatility = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=1).sum().values
    volatility = np.concatenate([np.zeros(9), volatility[9:]])  # align
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing Constants
    fastest_sc = 2 / (2 + 1)
    slowest_sc = 2 / (30 + 1)
    sc = (er * (fastest_sc - slowest_sc) + slowest_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 6h Bollinger Bands(20,2) for mean reversion regime
    bb_period = 20
    bb_std = 2
    ma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = ma_20 + (bb_std * std_20)
    bb_lower = ma_20 - (bb_std * std_20)
    
    # 6h ATR(14) for trend regime stops
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_regime = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all calculations
    start_idx = max(30, 14, 20, 10)  # ADX, ATR, BB, KAMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(kama[i]) or np.isnan(atr[i]) or 
            np.isnan(ma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx_aligned[i]
        kama_val = kama[i]
        atr_val = atr[i]
        bb_up = bb_upper[i]
        bb_low = bb_lower[i]
        vol_ok = vol_regime[i]
        
        if position == 0:
            # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
            if adx_val > 25:  # Trending regime
                # Trend following: enter when price extends beyond KAMA by 0.5*ATR
                long_signal = (close[i] > kama_val + 0.5 * atr_val) and vol_ok
                short_signal = (close[i] < kama_val - 0.5 * atr_val) and vol_ok
            elif adx_val < 20:  # Ranging regime
                # Mean reversion: fade at Bollinger Bands
                long_signal = (close[i] <= bb_low) and vol_ok
                short_signal = (close[i] >= bb_up) and vol_ok
            else:  # Transition regime (20 <= ADX <= 25) - no trading
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Regime change to ranging (ADX < 20) - take profit
            # 2. Price returns to KAMA (mean reversion exit)
            # 3. Opposite signal in same regime
            if adx_val < 20 or close[i] <= kama_val or \
               (adx_val > 25 and close[i] < kama_val - 0.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Regime change to ranging (ADX < 20) - take profit
            # 2. Price returns to KAMA (mean reversion exit)
            # 3. Opposite signal in same regime
            if adx_val < 20 or close[i] >= kama_val or \
               (adx_val > 25 and close[i] > kama_val + 0.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_KAMA_Regime_Adaptive_Consensus"
timeframe = "6h"
leverage = 1.0