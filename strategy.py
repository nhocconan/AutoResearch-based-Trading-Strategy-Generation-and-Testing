#!/usr/bin/env python3
"""
6h_ADX_Alligator_Trend_12hVolatilityFilter
Hypothesis: Combines ADX trend strength (12h) with Williams Alligator (6h) and 12h volatility regime filter.
In trending markets (ADX>25), Alligator alignment (jaw-teeth-lips) gives entry signals.
Volatility filter avoids whipsaw in low-vol regimes. Works in bull/bear: ADX captures trend strength regardless of direction.
Discrete sizing (0.25) limits drawdown and reduces fee churn. Targets 50-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 12h data for ADX and volatility filter
    df_12h = get_htf_data(prices, '12h')
    
    # === 12h ADX (trend strength) ===
    # True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = df_12h['high'].diff().values
    down_move = -df_12h['low'].diff().values
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR and DM
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 12h Volatility Regime Filter (ATR ratio) ===
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # >1 = expanding vol, <1 = contracting vol
    atr_ratio_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio)
    
    # === 6h Williams Alligator ===
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(arr, np.nan, dtype=float)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(arr)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
        return smma_vals
    
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines to 6h (no extra delay needed - SMMA is not lagging in confirmation sense)
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25  # 25% position
    
    # Warmup: need ADX(14+14), ATR(50), Alligator(13)
    start_idx = max(50, 50, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        vol_ratio = atr_ratio_aligned[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        close_val = close[i]
        
        # Trend strength filter: ADX > 25
        # Volatility filter: avoid low volatility chop (ATR ratio < 0.8) and extreme volatility (ATR ratio > 2.0)
        trend_filter = adx_val > 25.0
        vol_filter = (vol_ratio >= 0.8) and (vol_ratio <= 2.0)
        
        if position == 0:
            # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
            # Lips < Teeth < Jaw = bearish alignment
            bullish_align = lips_val > teeth_val and teeth_val > jaw_val
            bearish_align = lips_val < teeth_val and teeth_val < jaw_val
            
            if trend_filter and vol_filter and bullish_align:
                signals[i] = size
                position = 1
            elif trend_filter and vol_filter and bearish_align:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR ADX weakens
            if not (lips_val > teeth_val and teeth_val > jaw_val) or adx_val < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR ADX weakens
            if not (lips_val < teeth_val and teeth_val < jaw_val) or adx_val < 20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_Alligator_Trend_12hVolatilityFilter"
timeframe = "6h"
leverage = 1.0