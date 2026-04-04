#!/usr/bin/env python3
"""
Experiment #3047: 6h Williams %R + 1d ADX Regime Filter + Volume Spike
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe. 
ADX(14) from 1d timeframe filters for trending vs ranging regimes: only take mean-reversion 
trades when ADX < 25 (ranging market) and avoid trend-following signals when ADX >= 25. 
Volume spike (>1.8x 20-period average) confirms momentum exhaustion. This combines 
contrarian mean reversion with regime awareness to work in both bull (buy dips in range) 
and bear (sell rallies in range) markets while avoiding false signals during strong trends.
Target: 75-150 total trades over 4 years (19-37/year). Discrete position sizing: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3047_6h_williamsr_1d_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothed DM
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (ADX < 25)
        if adx_1d_aligned[i] >= 25:
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Williams %R mean reversion signals
            if williams_r[i] <= -80:  # Oversold -> long
                signals[i] = SIZE
            elif williams_r[i] >= -20:  # Overbought -> short
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals