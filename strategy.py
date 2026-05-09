#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Trix_Volume_Spike_Chop_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and Chop index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 1d: EMA12(EMA12(EMA12(close)))
    close_1d = pd.Series(df_1d['close'].values)
    ema1 = close_1d.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3.pct_change()).values
    
    # Smooth TRIX with 9-period EMA (signal line)
    trix_series = pd.Series(trix_raw)
    trix_signal = trix_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_signal_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Chop index on 1d: 100 * log10(sum(atr14) / (max(high,n) - min(low,n))) / log10(n)
    # ATR(14) on 1d
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    range_hl = max_high - min_low
    chop = 100 * np.log10(sum_atr14 / range_hl) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 2x 30-period average (~15-day average for 4h)
    vol_series = pd.Series(volume)
    vol_ma30 = vol_series.rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma30[i]
        chop_high = chop_aligned[i] > 61.8  # ranging market
        chop_low = chop_aligned[i] < 38.2   # trending market
        
        if position == 0:
            # Long: TRIX crosses above signal line in choppy market with volume
            if trix_signal_aligned[i] > trix_signal_aligned[i-1] and chop_high and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line in choppy market with volume
            elif trix_signal_aligned[i] < trix_signal_aligned[i-1] and chop_high and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below signal line
            if trix_signal_aligned[i] < trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above signal line
            if trix_signal_aligned[i] > trix_signal_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals