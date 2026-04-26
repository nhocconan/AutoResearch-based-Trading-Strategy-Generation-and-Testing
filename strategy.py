#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATRRegime_VolumeSpike_v1
Hypothesis: On 4h timeframe, combining Camarilla R1/S1 breakouts with 1d ATR-based regime filter (low volatility = mean reversion, high volatility = trend) and volume confirmation reduces false breakouts in ranging markets while capturing strong trends. The ATR regime filter adapts to changing market conditions, making it effective in both bull and bear markets by avoiding breakouts during low-volatility squeezes and participating during high-volatility breakouts. Volume confirmation ensures breakouts have conviction. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF ATR regime and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.full_like(tr, np.nan)
    atr_1d[13] = np.mean(tr[1:14])  # seed with first 14 values
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # ATR percentile rank over 50 periods for regime detection
    atr_percentile = np.full_like(atr_1d, np.nan)
    for i in range(49, len(atr_1d)):
        window = atr_1d[i-49:i+1]
        if not np.any(np.isnan(window)):
            atr_percentile[i] = (np.sum(window <= atr_1d[i]) - 1) / 49 * 100
    
    # Regime: High volatility (trending) when ATR percentile > 60, Low volatility (choppy) when < 40
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Calculate 1d Camarilla levels (R1, S1) using previous 1d's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift to get previous day's OHLC for Camarilla calculation
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    camarilla_range = prev_high_1d - prev_low_1d
    r1 = prev_close_1d + 1.1 * camarilla_range / 12
    s1 = prev_close_1d - 1.1 * camarilla_range / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for ATR percentile and volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_regime_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        volume_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakout_s1 = close[i] < s1_aligned[i]
        
        # Regime filter: Only trade in high volatility (trending) regimes
        high_vol_regime = atr_regime_aligned[i] > 60
        
        # Long logic: breakout above R1 in high volatility regime with volume
        if high_vol_regime and volume_spike and breakout_r1:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: breakout below S1 in high volatility regime with volume
        elif high_vol_regime and volume_spike and breakout_s1:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: exit when volatility drops (regime change) or opposite breakout
        elif position == 1 and (atr_regime_aligned[i] < 40 or breakout_s1):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (atr_regime_aligned[i] < 40 or breakout_r1):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0