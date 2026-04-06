#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 12h data for TRIX and regime filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # TRIX (15-period) on 12h close
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema1 = ema(close_12h, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    
    # TRIX calculation: (EMA3 - previous EMA3) / previous EMA3 * 100
    trix_raw = np.full_like(ema3, np.nan)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = ema(trix_raw, 9)
    
    # Align TRIX and signal to 4h
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix_raw)
    trix_signal_aligned = align_htf_to_ltf(prices, df_12h, trix_signal)
    
    # Volume spike detection on 12h (current volume > 2x average of last 20)
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > vol_ma_12h_aligned * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trix_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: TRIX crosses below signal or stoploss hit
            if (trix_aligned[i] < trix_signal_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TRIX crosses above signal or stoploss hit
            if (trix_aligned[i] > trix_signal_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: TRIX crosses above signal with volume spike
            if (trix_aligned[i] > trix_signal_aligned[i] and 
                trix_aligned[i-1] <= trix_signal_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TRIX crosses below signal with volume spike
            elif (trix_aligned[i] < trix_signal_aligned[i] and 
                  trix_aligned[i-1] >= trix_signal_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals