#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Regime
# Hypothesis: TRIX (12) captures momentum; a spike above zero with volume confirms institutional buying,
# while a spike below zero with volume confirms selling. Trades only in trending regimes (ADX > 25)
# to avoid whipsaws in ranging markets. Works in bull/bear by following momentum with volume confirmation.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_TRIX_Volume_Spike_Regime"
timeframe = "4h"
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
    
    # Get 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first value
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        up_move[0] = 0
        down_move[0] = 0
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        def smoothed_avg(arr, period):
            res = np.full_like(arr, np.nan)
            if len(arr) >= period:
                # First value is simple average
                res[period-1] = np.mean(arr[:period])
                # Subsequent values: Wilder smoothing
                for i in range(period, len(arr)):
                    res[i] = (res[i-1] * (period-1) + arr[i]) / period
            return res
        
        atr = smoothed_avg(tr, period)
        plus_di = 100 * smoothed_avg(plus_dm, period) / atr
        minus_di = 100 * smoothed_avg(minus_dm, period) / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = smoothed_avg(dx, period)
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate TRIX(12) on 4h close
    def calculate_trix(arr, period=12):
        # Triple EMA
        ema1 = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema2 = pd.Series(ema1).ewm(span=period, adjust=False, min_periods=period).mean().values
        ema3 = pd.Series(ema2).ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # TRIX = 100 * (EMA3 - prev EMA3) / prev EMA3
        trix = np.full_like(arr, np.nan)
        trix[period:] = 100 * (ema3[period:] - ema3[period-1:-1]) / ema3[period-1:-1]
        return trix
    
    trix_12 = calculate_trix(close, 12)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20) + 5
    
    for i in range(start_idx, n):
        if np.isnan(adx_14_1d_aligned[i]) or np.isnan(trix_12[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # Regime filter: trending market (ADX > 25)
        trending = adx_14_1d_aligned[i] > 25
        
        if position == 0:
            # Long: TRIX crosses above zero with volume in trending market
            if trix_12[i] > 0 and trix_12[i-1] <= 0 and volume_confirm and trending:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume in trending market
            elif trix_12[i] < 0 and trix_12[i-1] >= 0 and volume_confirm and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TRIX crosses below zero
            if trix_12[i] < 0 and trix_12[i-1] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TRIX crosses above zero
            if trix_12[i] > 0 and trix_12[i-1] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals