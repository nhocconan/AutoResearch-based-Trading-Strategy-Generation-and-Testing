#!/usr/bin/env python3
"""
6h_RSI_Pullback_Trend_1d
Hypothesis: RSI(14) pullback in trending markets with 1-day EMA50 trend filter and volume confirmation.
In uptrend (price > 1d EMA50), long when RSI crosses above 40 from below with volume.
In downtrend (price < 1d EMA50), short when RSI crosses below 60 from above with volume.
Aims to capture trend continuations during pullbacks with reduced false signals in range markets.
Target: 10-30 trades per year (~40-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_Pullback_Trend_1d"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: RSI crosses above 40 from below in uptrend regime + volume
            long_entry = (rsi[i] > 40) and (rsi[i-1] <= 40) and uptrend_regime and volume_confirm
            # Short: RSI crosses below 60 from above in downtrend regime + volume
            short_entry = (rsi[i] < 60) and (rsi[i-1] >= 60) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 70 (overbought) or trend changes to downtrend
            if (rsi[i] >= 70) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 30 (oversold) or trend changes to uptrend
            if (rsi[i] <= 30) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals