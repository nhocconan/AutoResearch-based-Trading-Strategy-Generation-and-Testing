#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_v1
Hypothesis: TRIX zero-cross with volume confirmation captures momentum shifts with low trade frequency (~20-35 trades/year). Works in bull/bear by taking long on zero-cross above, short on zero-cross below. Uses 1d ATR filter to avoid choppy markets. Discrete sizing (0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # TRIX(12,9,9) on 4h close - triple EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average (stricter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of TRIX calc (12+9+9=30), volume MA (20), ATR (14)
    start_idx = max(30, 20, 14)
    
    for i in range(start_idx, n):
        trix_val = trix[i]
        trix_sig_val = trix_signal[i]
        atr_val = atr_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Skip if any data not ready
        if (np.isnan(trix_val) or np.isnan(trix_sig_val) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # ATR filter: avoid extremely low volatility (chop) and extremely high volatility (panic)
        # Use 1d ATR relative to its 50-period mean
        if i >= start_idx + 36:  # Need history for ATR mean
            atr_ma = np.nanmean(atr_1d_aligned[max(0, i-50):i])
            if atr_ma > 0:
                atr_ratio = atr_val / atr_ma
                # Only trade when ATR ratio between 0.5 and 2.0 (avoid extremes)
                vol_filter = (atr_ratio >= 0.5) and (atr_ratio <= 2.0)
            else:
                vol_filter = True
        else:
            vol_filter = True
        
        # Entry conditions: TRIX crosses above/below signal line + volume + vol filter
        long_condition = (trix_val > trix_sig_val) and (trix[i-1] <= trix_signal[i-1]) and vol_conf and vol_filter
        short_condition = (trix_val < trix_sig_val) and (trix[i-1] >= trix_signal[i-1]) and vol_conf and vol_filter
        
        # Exit conditions: opposite cross
        long_exit = (position == 1 and (trix_val < trix_sig_val))
        short_exit = (position == -1 and (trix_val > trix_sig_val))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0