#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and daily ATR filter.
Uses daily ATR for volatility filter (only trade when ATR > 20-period average) and
volume confirmation to avoid false breakouts. In ranging markets (low ATR), stay flat.
Target: 20-50 trades per year (~80-200 over 4 years) to minimize fee drag.
Works in both bull/bear by following breakout direction with volatility filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_daily_atr_volume_v1"
timeframe = "4h"
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
    
    # === DAILY ATR FILTER (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate True Range and ATR
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # True Range: max(high-low, |high-prev_close|, |low-prev_close|)
    tr1 = d_high - d_low
    tr2 = np.abs(d_high - np.roll(d_close, 1))
    tr3 = np.abs(d_low - np.roll(d_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = d_high[0] - d_low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR and its MA to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # === 4H DONCHIAN CHANNELS (LTF) ===
    donchian_len = 20
    upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # === VOLUME CONFIRMATION (LTF) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_len, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr_ma_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > MA(ATR)
        vol_filter = atr_aligned[i] > atr_ma_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian band OR volatility filter fails
            if close[i] <= lower[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian band OR volatility filter fails
            if close[i] >= upper[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation and volatility filter
            if volume[i] <= vol_ma[i] or not vol_filter:
                signals[i] = 0.0
                continue
            
            # Breakout entry
            if close[i] > upper[i]:  # Break above upper band -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lower[i]:  # Break below lower band -> short
                position = -1
                signals[i] = -0.25
    
    return signals