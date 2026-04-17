#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour price crossing 200-day EMA with volume confirmation and ATR-based stop.
# The 200-day EMA provides a robust trend filter that works in both bull and bear markets.
# Price crossing above/below the 200-day EMA with volume confirmation captures strong momentum shifts.
# ATR-based stops limit drawdowns during volatile periods. Target: 25-35 trades/year (100-140 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA200, ATR, and volume average ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA200 on daily data
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # ATR calculation on daily data (14-period)
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        if len(high) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 20-day average volume on daily data
    volume_1d_series = pd.Series(volume_1d)
    vol_avg20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_avg20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg20_1d)
    
    signals = np.zeros(n)
    position = 0
    warmup = 200  # Sufficient for EMA200
    
    for i in range(warmup, n):
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_avg20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        vol_filter = vol_1d_current > 1.5 * vol_avg20_1d_aligned[i]
        
        if position == 0:
            # Long: price crosses above 200-day EMA + volume confirmation
            if close[i] > ema200_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below 200-day EMA + volume confirmation
            elif close[i] < ema200_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 200-day EMA
            if close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 200-day EMA
            if close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA200_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0