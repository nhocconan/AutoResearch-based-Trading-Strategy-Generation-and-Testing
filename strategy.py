#!/usr/bin/env python3
name = "1D_Weekly_HMA_Trend_Filter_Strategy"
timeframe = "1d"
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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly HMA (Hull Moving Average) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, n + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        half_n = n // 2
        sqrt_n = int(np.sqrt(n))
        wma_half = wma(arr, half_n)
        wma_full = wma(arr, n)
        # Pad wma_half to same length as wma_full for subtraction
        wma_half_padded = np.full_like(wma_full, np.nan)
        wma_half_padded[-len(wma_half):] = wma_half
        raw_hma = 2 * wma_half_padded - wma_full
        return wma(raw_hma, sqrt_n)
    
    # Calculate weekly HMA(20)
    hma_20_1w = hma(df_1w['close'].values, 20)
    hma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_20_1w)
    
    # Daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.003 * close  # ATR > 0.3% of price
    
    # Volume filter: current volume > 1.5x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08:00 - 20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(hma_20_1w_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (1.5x average volume)
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price above weekly HMA + volume spike
            if (close[i] > hma_20_1w_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly HMA + volume spike
            elif (close[i] < hma_20_1w_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price crosses back over weekly HMA
            if (position == 1 and close[i] < hma_20_1w_aligned[i]) or \
               (position == -1 and close[i] > hma_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals