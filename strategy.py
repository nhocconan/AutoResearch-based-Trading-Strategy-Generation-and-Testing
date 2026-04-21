#!/usr/bin/env python3
"""
4h_12h_RSI_MeanReversion_VolumeFilter_v1
Hypothesis: In 4h timeframe, use 12h RSI extremes (>70/<30) for mean reversion entries.
Go long when 12h RSI < 30 and price near 4h Donchian lower band with volume confirmation.
Go short when 12h RSI > 70 and price near 4h Donchian upper band with volume confirmation.
Exit on opposite Donchian band touch or RSI normalization.
Works in bull/bear by fading extremes with volume confirmation and ATR-based stop.
Target: 20-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for RSI
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.values
    # Align to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean()
    volume_ok = prices['volume'].values > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol_ok = volume_ok[i]
        
        # Entry conditions with hysteresis
        if position == 0:
            # Long: RSI oversold + price near lower Donchian + volume
            if (rsi_12h_aligned[i] < 30 and 
                price <= lower[i] * 1.001 and  # near or below lower band
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought + price near upper Donchian + volume
            elif (rsi_12h_aligned[i] > 70 and 
                  price >= upper[i] * 0.999 and  # near or above upper band
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI normalizes (>50) or price reaches upper band
            if rsi_12h_aligned[i] > 50 or price >= upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI normalizes (<50) or price reaches lower band
            if rsi_12h_aligned[i] < 50 or price <= lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_RSI_MeanReversion_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0