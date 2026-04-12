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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (S3, S4 for support; R3, R4 for resistance)
    # S3 = close - (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1
    # R3 = close + (high - low) * 1.1/2
    # R4 = close + (high - low) * 1.1
    s3 = close_1d - range_1d * 1.1 / 2
    s4 = close_1d - range_1d * 1.1
    r3 = close_1d + range_1d * 1.1 / 2
    r4 = close_1d + range_1d * 1.1
    
    # Align to lower timeframe (4h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    
    # Calculate 15-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr15 = np.full(n, np.nan)
    for i in range(14, n):
        atr15[i] = np.nanmean(tr[i-14:i+1])
    
    # Calculate 30-period ATR EMA for volatility regime
    atr_ema30 = np.full(n, np.nan)
    atr_series = pd.Series(atr15)
    atr_ema30_values = atr_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    atr_ema30[:] = atr_ema30_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(atr15[i]) or 
            np.isnan(atr_ema30[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR15 > 1.2x 30-period ATR EMA (elevated volatility)
        vol_filter = atr15[i] > atr_ema30[i] * 1.2
        
        # Entry conditions: 
        # Long when price touches S3/S4 with volume confirmation
        # Short when price touches R3/R4 with volume confirmation
        long_entry = ((close[i] <= s3_aligned[i] * 1.002) or (close[i] <= s4_aligned[i] * 1.002)) and vol_filter
        short_entry = ((close[i] >= r3_aligned[i] * 0.998) or (close[i] >= r4_aligned[i] * 0.998)) and vol_filter
        
        # Exit conditions: price returns to pivot or volatility drops
        long_exit = (close[i] >= pivot_aligned[i] * 0.998) or (atr15[i] < atr_ema30[i] * 0.8)
        short_exit = (close[i] <= pivot_aligned[i] * 1.002) or (atr15[i] < atr_ema30[i] * 0.8)
        
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

name = "4h_1d_camarilla_pivot_touch_vol_filter_v1"
timeframe = "4h"
leverage = 1.0