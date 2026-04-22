#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Chop Regime Filter.
Long when price breaks above Camarilla R3 with volume spike and chop > 61.8 (range).
Short when price breaks below Camarilla S3 with volume spike and chop > 61.8 (range).
Exit when price crosses Camarilla H4/L4 or chop < 38.2 (trending).
Camarilla levels provide institutional support/resistance; volume confirms participation;
chop filter avoids trending markets where pivot breaks fail. Designed for 12h timeframe
to capture major reversals in ranging markets (common in BTC/ETH 2025 bear/range).
"""

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
    
    # Calculate pivot points from previous day (need daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    R4 = prev_close + (prev_high - prev_low) * 1.5000
    R3 = prev_close + (prev_high - prev_low) * 1.2500
    R2 = prev_close + (prev_high - prev_low) * 1.1666
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    PP = (prev_high + prev_low + prev_close) / 3
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    S2 = prev_close - (prev_high - prev_low) * 1.1666
    S3 = prev_close - (prev_high - prev_low) * 1.2500
    S4 = prev_close - (prev_high - prev_low) * 1.5000
    
    # Align Camarilla levels to 12h timeframe (they change only at daily open)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    H4_12h = align_htf_to_ltf(prices, df_1d, R4)
    L4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Chopiness Index (14-period) to detect ranging markets
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_sum14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum14 / (atr14 * 14)) / np.log10(14)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(H4_12h[i]) or 
            np.isnan(L4_12h[i]) or np.isnan(chop[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and chop > 61.8 (ranging market)
            if close[i] > R3_12h[i] and vol_spike and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike and chop > 61.8 (ranging market)
            elif close[i] < S3_12h[i] and vol_spike and chop[i] > 61.8:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses H4/L4 or chop < 38.2 (trending market)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below H4 or market starts trending
                if close[i] < H4_12h[i] or chop[i] < 38.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above L4 or market starts trending
                if close[i] > L4_12h[i] or chop[i] < 38.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Volume_ChopFilter"
timeframe = "12h"
leverage = 1.0