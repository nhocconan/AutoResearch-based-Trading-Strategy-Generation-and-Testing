#!/usr/bin/env python3
"""
4h_atr_breakout_1d_trend_volume_v5
Hypothesis: ATR-based breakouts from daily ATR-multiplied ranges, filtered by daily EMA trend and volume spikes, work in both bull and bear markets by adapting to volatility.
Only trade when volatility expands (ATR ratio > 1.0) and price breaks beyond daily ATR bands with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v5"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ATR, EMA, and range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily ATR (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First day
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily EMA for trend filter
    close_series = pd.Series(close_1d)
    ema_1d = close_series.ewm(span=20, min_periods=20).mean().values
    
    # Average True Range for volatility regime (20-day)
    atr_ma = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio (current vs average) - volatility filter
    atr_ratio = np.where(atr_ma > 0, atr_1d / atr_ma, 1.0)
    
    # Daily range bands: EMA ± (ATR * multiplier)
    multiplier = 1.5
    upper_band = ema_1d + (atr_1d * multiplier)
    lower_band = ema_1d - (atr_1d * multiplier)
    
    # Align to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume confirmation: volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR ratio > 1.0 (expanding volatility)
        vol_filter = atr_ratio_aligned[i] > 1.0
        
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below EMA (trend change)
            if close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above EMA (trend change)
            if close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band with volume and volatility expansion
            if close[i] > upper_band_aligned[i] and vol_confirmed and vol_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band with volume and volatility expansion
            elif close[i] < lower_band_aligned[i] and vol_confirmed and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals