#!/usr/bin/env python3
"""
Hypothesis: 1h strategy using 4h trend (EMA50) and 1d regime filter (ATR ratio) with volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for EMA50 trend direction, 1d for ATR regime filter (trending vs choppy) and volume spike.
- Entry: Long when price > 4h EMA50 AND ATR ratio > 1.2 (trending) AND volume > 2.0 * 20-period average volume (1d).
         Short when price < 4h EMA50 AND ATR ratio > 1.2 (trending) AND volume > 2.0 * 20-period average volume (1d).
- Exit: Opposite condition (price crosses back over 4h EMA50) or regime turns choppy (ATR ratio < 0.8).
- Signal size: 0.20 discrete to minimize fee drag.
- Uses 4h/1d for signal direction/regime, 1h only for entry timing to reduce trade frequency.
- Works in both bull and bear markets by only trading in trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(10) and ATR(30) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ATR30
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # Align length
    
    # ATR(10) and ATR(30)
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio for regime: >1.2 = trending, <0.8 = choppy
    atr_ratio = atr10 / atr30
    
    # Align ATR ratio to 1h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Session filter: 08-20 UTC (active trading hours)
    # open_time is already datetime64[ns], so we can use .hour directly via index
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # Need 50 for EMA50, 30 for ATR30
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade in trending markets (ATR ratio > 1.2)
        trending_regime = atr_ratio_aligned[i] > 1.2
        choppy_regime = atr_ratio_aligned[i] < 0.8
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit if: price crosses EMA50 OR regime turns choppy
            exit_condition = False
            if position == 1:  # Long
                if curr_close < ema_50_4h_aligned[i] or choppy_regime:
                    exit_condition = True
            elif position == -1:  # Short
                if curr_close > ema_50_4h_aligned[i] or choppy_regime:
                    exit_condition = True
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: only in trending regime with volume confirmation
        if position == 0 and trending_regime and volume_confirm:
            # Long: price > 4h EMA50
            if curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50
            elif curr_close < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_EMA50_Trend_1dATRRegime_VolumeConfirm_Session_v1"
timeframe = "1h"
leverage = 1.0