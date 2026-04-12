#!/usr/bin/env python3
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points using previous day's data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point and support/resistance levels (Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Key levels: S1, R1 (Camarilla) - tighter range for fewer trades
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    
    # Align levels to 12h timeframe
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume filter: 20-period EMA
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    # ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-14:i+1])
    
    # Additional ATR filter: current ATR > 50% of 20-period ATR mean
    atr_ma = np.full(n, np.nan)
    if n >= 34:
        atr_series = pd.Series(atr)
        atr_ma_values = atr_series.rolling(window=20, min_periods=20).mean().values
        atr_ma[:] = atr_ma_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(s1_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(pivot_12h[i]) or np.isnan(vol_ema[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.5
        
        # Volatility filter: ATR > 0.5 * 20-period ATR mean
        vol_filter = atr[i] > atr_ma[i] * 0.5
        
        # Entry conditions: Touch of S1/R1 with volume and volatility (mean reversion)
        long_entry = (low[i] <= s1_12h[i]) and volume_filter and vol_filter
        short_entry = (high[i] >= r1_12h[i]) and volume_filter and vol_filter
        
        # Exit conditions: Return to pivot
        long_exit = close[i] > pivot_12h[i]
        short_exit = close[i] < pivot_12h[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_s1r1_mean_reversion_vol_filter_v2"
timeframe = "12h"
leverage = 1.0