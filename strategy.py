#!/usr/bin/env python3
"""
1h_price_position_4h1d_trend_volume_v1
Hypothesis: Use price position within 4h ATR-based channel for mean reversion in ranging markets,
filtered by 1d trend direction. Enter when price reaches channel extremes (oversold/overbought)
with volume confirmation, following the 1d trend. Works in bull/bear by aligning with higher timeframe trend.
Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_price_position_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for channel calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h ATR(14) for channel width
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h SMA(20) for channel center
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    
    # Channel bounds: SMA ± ATR
    upper_channel = sma_20 + atr_14
    lower_channel = sma_20 - atr_14
    
    # Align 4h channel to 1h
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    sma_aligned = align_htf_to_ltf(prices, df_4h, sma_20)
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean()
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50.values)
    
    # Volume confirmation (24-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(sma_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to SMA or breaks below lower channel
            if close[i] >= sma_aligned[i] or close[i] <= lower_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: price returns to SMA or breaks above upper channel
            if close[i] <= sma_aligned[i] or close[i] >= upper_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: price at lower channel with volume and uptrend (price > EMA50)
            if (close[i] <= lower_aligned[i] and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: price at upper channel with volume and downtrend (price < EMA50)
            elif (close[i] >= upper_aligned[i] and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals