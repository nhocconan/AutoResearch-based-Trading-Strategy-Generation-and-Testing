#!/usr/bin/env python3
# 4h_VolumeWeighted_Donchian_Breakout
# Hypothesis: Donchian breakouts with volume-weighted momentum and volatility filtering
# capture sustained moves. Uses 1d trend filter (close > SMA50) for long-only bias in bull markets,
# with mean-reversion shorts in bear markets via RSI extremes. Volume confirmation ensures
# institutional participation. Designed for low trade frequency (<30/year) to minimize fee drag.

name = "4h_VolumeWeighted_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend filter
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate 1d RSI(14) for mean-reversion signals
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values, additional_delay_bars=2)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(high_4h, np.nan)
    
    for i in range(len(high_4h)):
        if i >= 19:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Align Donchian levels to LTF
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1]) if len(close_4h) > 1 else np.array([])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1]) if len(close_4h) > 1 else np.array([])
    close_4h = df_4h['close'].values
    if len(high_4h) > 1:
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    else:
        tr = np.array([np.nan])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure indicators are calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(sma50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Donchian breakout + uptrend (price > SMA50) + volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > sma50_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown + overbought RSI + volume (mean reversion in bear)
            elif (close[i] < donchian_low_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend breaks
            if close[i] < donchian_low_aligned[i] or close[i] < sma50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or RSI normalizes
            if close[i] > donchian_high_aligned[i] or rsi_1d_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals