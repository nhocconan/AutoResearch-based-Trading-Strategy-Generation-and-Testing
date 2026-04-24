#!/usr/bin/env python3
"""
Hypothesis: 1h momentum with 4h/1d trend filter and volume confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h for trend direction (EMA50 slope), 1d for regime filter (ATR ratio) and volume spike.
- Entry: Long when 1h price > 1h EMA20 AND 4h EMA50 rising AND 1d ATR ratio > 1.2 (trending) AND 1h volume > 1.5 * 20-period average volume.
         Short when 1h price < 1h EMA20 AND 4h EMA50 falling AND 1d ATR ratio > 1.2 AND 1h volume > 1.5 * 20-period average volume.
- Exit: Opposite 1h EMA20 cross.
- Signal size: 0.20 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading momentum in HTF-trending regimes, avoiding whipsaws in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA20 for entry signal
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 4h EMA50 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 4h EMA50 slope (rising/falling)
    ema50_slope = np.diff(ema50_4h_aligned, prepend=ema50_4h_aligned[0])
    ema50_rising = ema50_slope > 0
    ema50_falling = ema50_slope < 0
    
    # Calculate 1d ATR ratio for regime filter
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
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 1h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 30)  # EMA20, EMA50, ATR30
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(ema20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: only trade momentum in trending markets (ATR ratio > 1.2)
        trending_regime = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite 1h EMA20 cross
        if position != 0:
            # Exit long: price < EMA20
            if position == 1:
                if curr_close < ema20[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > EMA20
            elif position == -1:
                if curr_close > ema20[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: 1h EMA20 cross with HTF trend and volume filters
        if position == 0:
            # Long: price > EMA20 AND 4h EMA50 rising AND trending regime AND volume confirmation
            long_condition = (curr_close > ema20[i] and 
                            ema50_rising[i] and
                            trending_regime and
                            volume_confirm)
            
            # Short: price < EMA20 AND 4h EMA50 falling AND trending regime AND volume confirmation
            short_condition = (curr_close < ema20[i] and 
                             ema50_falling[i] and
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.20
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.20
    
    return signals

name = "1h_EMA20_Momentum_4hEMA50Trend_1dATRRegime_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0