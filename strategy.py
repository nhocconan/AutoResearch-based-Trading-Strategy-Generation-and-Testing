#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX + Williams Alligator trend alignment with 12h volume confirmation
# ADX > 25 confirms strong trend, Alligator lines aligned (Lips>Teeth>Jaw for long, Jaw>Teeth>Lips for short)
# 12h volume > 1.5x 20-period average confirms institutional participation
# Works in bull/bear by following strong trends, avoids chop via ADX filter
# Target: 20-35 trades/year per symbol
name = "6h_ADX_Alligator_12hVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume average for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).sum().values / (atr * period + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Williams Alligator components (SMMA = Smoothed Moving Average)
    def smoothed_moving_average(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close, 13)  # Blue line (13-period)
    teeth = smoothed_moving_average(close, 8)   # Red line (8-period)
    lips = smoothed_moving_average(close, 5)    # Green line (5-period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + Lips > Teeth > Jaw (bullish alignment) + volume confirmation
            if (adx[i] > 25 and 
                lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                volume[i] > (vol_ma_12h_aligned[i] * 1.5)):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (strong trend) + Jaw > Teeth > Lips (bearish alignment) + volume confirmation
            elif (adx[i] > 25 and 
                  jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  volume[i] > (vol_ma_12h_aligned[i] * 1.5)):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if trend weakens (ADX < 20) or Alligator lines intertwine
            if (adx[i] < 20) or (lips[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if trend weakens (ADX < 20) or Alligator lines intertwine
            if (adx[i] < 20) or (jaw[i] < teeth[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals