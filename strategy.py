#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dHMA21_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 or below S1 levels from prior 1d session captures institutional breakouts. Combined with 1d HMA21 trend filter (lag-reducing), volume spike confirmation, and ATR-based stoploss. Designed for moderate trade frequency (~30-50/year) to minimize fee drag and work in both bull (breakout continuation) and bear (breakdown continuation) regimes. Uses HMA instead of EMA for better trend responsiveness with less lag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    # WMA helper
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan, dtype=float)
        weights = np.arange(1, window + 1)
        wma_vals = np.full_like(values, np.nan, dtype=float)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(arr, half)
    wma_full = wma(arr, period)
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt)
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels and HMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # === 1-day HMA21 for trend filter ===
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R2, S2
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    camarilla_s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === ATR for stoploss (14-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        hma_21 = hma_21_1d_aligned[i]
        vol_spike = vol_ratio[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above R1 (bullish breakout) + above 1d HMA21 + volume spike > 1.5
            if price_close > r1 and price_close > hma_21 and vol_spike > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S1 (bearish breakdown) + below 1d HMA21 + volume spike > 1.5
            elif price_close < s1 and price_close < hma_21 and vol_spike > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2 * ATR from entry
            if position == 1:
                if price_close < entry_price - 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0