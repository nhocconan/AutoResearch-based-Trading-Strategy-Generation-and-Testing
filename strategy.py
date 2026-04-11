#!/usr/bin/env python3
# 4h_1d_kama_volatility_regime_v1
# Strategy: 4h KAMA trend + 1d volatility regime filter (ATR ratio) + volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in chop. 1d ATR ratio identifies high/low volatility regimes.
# In high volatility (trending), follow KAMA direction. In low volatility (choppy), avoid trades. Volume confirms institutional participation.
# Designed for low trade frequency (<50/year) to minimize fee drag. Works in bull via KAMA uptrends and bear via KAMA downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_volatility_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h KAMA (ER=10, fast=2, slow=30)
    def kama(close, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_values = kama(close, 10, 2, 30)
    
    # 1d ATR ratio (current ATR / 50-period average ATR) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i == 0:
            atr[i] = tr[i]
        else:
            atr[i] = (atr[i-1] * 29 + tr[i]) / 30  # Wilder smoothing
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(kama_values[i]) or np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility regime filter: only trade in high volatility (trending) markets
        # ATR ratio > 1.2 indicates elevated volatility (trending regime)
        volatility_regime = atr_ratio_aligned[i] > 1.2
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # KAMA trend direction
        kama_bullish = close[i] > kama_values[i]
        kama_bearish = close[i] < kama_values[i]
        
        # Entry conditions
        # Long: KAMA bullish AND high volatility regime AND volume confirmation
        if kama_bullish and volatility_regime and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA bearish AND high volatility regime AND volume confirmation
        elif kama_bearish and volatility_regime and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Low volatility regime (choppy market) or opposite KAMA signal
        elif position == 1 and (not volatility_regime or not kama_bullish):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not volatility_regime or not kama_bearish):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals