#!/usr/bin/env python3
"""
6h Williams %R Alligator + 1d EMA50 Trend Filter
- Williams %R(14) from Alligator: long when %R crosses above -80 (exit oversold), short when crosses below -20 (exit overbought)
- Alligator filter: only trade when Jaw (13) > Teeth (8) > Lips (5) for bullish alignment OR reverse for bearish
- 1d EMA50 trend filter: long only when 1d close > EMA50, short only when 1d close < EMA50
- Volume confirmation: current volume > 1.8 * 20-period average (moderate spike to avoid churn)
- Signal size: 0.25 discrete levels
- Designed to catch reversals in ranging markets while respecting higher-timeframe trend
- Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14): %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We want %R crossing above -80 (long) or below -20 (short)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Alligator lines (Smoothed Moving Average - using SMA as approximation)
    # Jaw: 13-period, Teeth: 8-period, Lips: 5-period
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Alligator alignment: bullish when Jaw > Teeth > Lips, bearish when Jaw < Teeth < Lips
    bullish_alligator = (jaw > teeth) & (teeth > lips)
    bearish_alligator = (jaw < teeth) & (teeth < lips)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Trend filter
    bullish_regime = close > ema_50_1d_aligned
    bearish_regime = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 8, 5, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND bullish Alligator AND bullish regime AND volume
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                bullish_alligator[i] and bullish_regime[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND bearish Alligator AND bearish regime AND volume
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  bearish_alligator[i] and bearish_regime[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum fading) OR Alligator alignment breaks
            if williams_r[i] < -50 or not bullish_alligator[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 OR Alligator alignment breaks
            if williams_r[i] > -50 or not bearish_alligator[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Alligator_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0