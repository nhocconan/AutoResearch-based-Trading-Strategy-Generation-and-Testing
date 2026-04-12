# 6h_1d_trix_volume_regime
# Hypothesis: TRIX momentum on 6h with volume confirmation and 1d regime filter (ADX > 25 for trending, < 20 for ranging).
# In trending markets (ADX > 25), TRIX crossovers signal continuation.
# In ranging markets (ADX < 20), TRIX extremes signal mean reversion.
# Volume confirms momentum strength. Target: 15-35 trades/year (60-140 total over 4 years).

name = "6h_1d_trix_volume_regime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for TRIX, volume MA, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # TRIX: 15-period EMA applied three times, then % change
    def ema(series, period):
        return pd.Series(series).ewm(span=period, adjust=False).mean().values
    
    ema1 = ema(close_1d, 15)
    ema2 = ema(ema1, 15)
    ema3 = ema(ema2, 15)
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth with Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[1:period])
        plus_di[period-1] = np.mean(plus_dm[1:period]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        minus_di[period-1] = np.mean(minus_dm[1:period]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Volume confirmation: 1d volume > 1.3x 20-day average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm_1d = volume_1d > (vol_ma_1d * 1.3)
    
    # Align indicators to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Volume confirmation on 6b
        vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm_6h = volume[i] > (vol_ma_6h[i] * 1.5) if not np.isnan(vol_ma_6h[i]) else False
        
        # Trading logic
        if is_trending and vol_confirm_6h:
            # Trending market: TRIX crossovers signal momentum
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and position != -1:
                position = -1
                signals[i] = -0.25
        elif is_ranging:
            # Ranging market: TRIX extremes signal mean reversion
            if trix_aligned[i] < -0.5 and trix_aligned[i-1] >= -0.5 and position != 1:
                position = 1
                signals[i] = 0.25
            elif trix_aligned[i] > 0.5 and trix_aligned[i-1] <= 0.5 and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        if position == 1:
            # Exit long: TRIX crosses below zero or reverses from extreme
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            elif trix_aligned[i] > 0.5 and trix_aligned[i-1] <= 0.5:  # Overbought in range
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: TRIX crosses above zero or reverses from extreme
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            elif trix_aligned[i] < -0.5 and trix_aligned[i-1] >= -0.5:  # Oversold in range
                position = 0
                signals[i] = 0.0
        
        # Hold position
        if position == 1 and signals[i] == 0.0:
            signals[i] = 0.25
        elif position == -1 and signals[i] == 0.0:
            signals[i] = -0.25
    
    return signals