#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Filter
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA50 trend filter on 4h timeframe. Uses volume confirmation (1.5x average) and avoids choppy markets (CHOP < 61.8). Designed for lower trade frequency (20-40/year) to reduce fee drag while capturing strong trending moves. Uses discrete position sizing (0.25) to minimize churn.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous 1d bar's high, low, close for Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels: R1, S1
    camarilla_range = prev_high - prev_low
    R1 = prev_close + camarilla_range * 1.0/12
    S1 = prev_close - camarilla_range * 1.0/12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 1.5x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for volatility (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_14 = atr  # reuse ATR calculation
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_raw = np.where((max_high_14 - min_low_14) <= 0, 100, chop_raw)
    chop_raw = np.where(np.isnan(chop_raw) | np.isinf(chop_raw), 50, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA (50), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_raw[i])):
            signals[i] = 0.0
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        chop_val = chop_raw[i]
        
        # Regime filter: only trade when not too choppy (CHOP < 61.8 = trending market)
        regime_filter = chop_val < 61.8
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA50), volume confirmation, good regime
            long_signal = (high_val > R1_val) and (close_val > ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val) and regime_filter
            # Short: break below S1, downtrend (close < EMA50), volume confirmation, good regime
            short_signal = (low_val < S1_val) and (close_val < ema_50_1d_val) and (volume_val > 1.5 * vol_ma_val) and regime_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal (close < EMA50) or regime becomes too choppy
            if (close_val < ema_50_1d_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal (close > EMA50) or regime becomes too choppy
            if (close_val > ema_50_1d_val) or (chop_val >= 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA50_Trend_Filter"
timeframe = "4h"
leverage = 1.0