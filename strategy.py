#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
- Uses 4h timeframe (primary) and 1d HTF for trend alignment (proven pattern from DB)
- Camarilla R3/S3 levels from previous 1d candle (structure-based breakout)
- Long when price breaks above R3 AND price > 1d EMA34 (uptrend) AND volume > 2.0 * volume MA(20)
- Short when price breaks below S3 AND price < 1d EMA34 (downtrend) AND volume > 2.0 * volume MA(20)
- Exit when price reverts to the 1d VWAP (mean reversion to fair value)
- Discrete signal size: 0.30 to balance return and fee drag
- Target: 75-200 total trades over 4 years (19-50/year) as per 4h timeframe recommendation
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
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
    
    # Calculate 1d EMA34 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d VWAP for exit (volume weighted average price)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (pd.Series(typical_price_1d * df_1d['volume'].values).cumsum() / 
               pd.Series(df_1d['volume'].values).cumsum()).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d candle
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align Camarilla levels to 4h timeframe (previous 1d levels available at open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * volume_ma)
    
    # Trend filter: price above/below 1d EMA34
    uptrend = close > ema_34_1d_aligned
    downtrend = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1d EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price reverts to 1d VWAP (fair value)
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price reverts to 1d VWAP (fair value)
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Camarilla_R3S3_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0