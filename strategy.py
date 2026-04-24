#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + ADX regime filter with volume confirmation.
- Primary timeframe: 6h, HTF: 1d for Elder Ray (Bull/Bear Power) and ADX.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
- Regime: ADX(14) > 25 = trending (use Elder Ray extremes), ADX <= 25 = ranging (fade at power extremes).
- Entry logic: 
    * Trending (ADX > 25): Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling.
    * Ranging (ADX <= 25): Long when Bear Power < -0.5 * ATR(10) and turning up, Short when Bull Power > 0.5 * ATR(10) and turning down.
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA (moderate to avoid overtrading).
- Discrete signal size: 0.25 to balance reward and risk, minimizing fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull/bear: regime filter adapts to market conditions, Elder Ray captures power shifts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d ATR(10) for ranging filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]  # avoid NaN on first
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1d = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    
    # Calculate 1d ADX(14) for regime filter
    # +DM, -DM, TR
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_atr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_atr[0] = high_1d[0] - low_1d[0]
    atr_14_1d = pd.Series(tr_atr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_1d
    minus_di_14_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_14_1d
    dx_14_1d = 100 * np.abs(plus_di_14_1d - minus_di_14_1d) / (plus_di_14_1d + minus_di_14_1d)
    adx_14_1d = pd.Series(dx_14_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 14)  # Need sufficient lookback for all
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(atr_10_1d_aligned[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime: ADX > 25 = trending, else ranging
        is_trending = adx_14_1d_aligned[i] > 25
        
        if position == 0:
            if is_trending:
                # Trending: long when bull power > 0 and rising, short when bear power < 0 and falling
                if bull_power[i] > 0 and bull_power[i] > bull_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif bear_power[i] < 0 and bear_power[i] < bear_power[i-1]:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging: long when bear power < -0.5 * ATR and turning up, short when bull power > 0.5 * ATR and turning down
                if bear_power[i] < (-0.5 * atr_10_1d_aligned[i]) and bear_power[i] > bear_power[i-1]:
                    signals[i] = 0.25
                    position = 1
                elif bull_power[i] > (0.5 * atr_10_1d_aligned[i]) and bull_power[i] < bull_power[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: power deteriorates or opposite signal
            if is_trending:
                if bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                if bear_power[i] >= (-0.5 * atr_10_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: power deteriorates or opposite signal
            if is_trending:
                if bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                if bull_power[i] <= (0.5 * atr_10_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0