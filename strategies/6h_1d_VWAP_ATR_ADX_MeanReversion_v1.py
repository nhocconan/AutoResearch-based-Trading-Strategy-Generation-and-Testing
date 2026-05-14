#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Volume Weighted Average Price (VWAP) deviation with
# 1-day Average True Range (ATR) normalization and 1-day ADX trend filter.
# Long when price deviates below VWAP by more than 1.0 * ATR(14) AND ADX > 25 (trending).
# Short when price deviates above VWAP by more than 1.0 * ATR(14) AND ADX > 25.
# Exit when price returns to VWAP or ADX drops below 20.
# VWAP acts as dynamic fair value; ATR normalization adapts to volatility.
# ADX filter ensures trading only in trending conditions, avoiding whipsaws in ranges.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP, ATR, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for ATR(14) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate VWAP (typical price * volume cumulative / volume cumulative)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Calculate ATR (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR = smoothed TR (using Wilder's smoothing: ATR today = (Prior ATR * 13 + Today TR) / 14)
    atr = np.full_like(tr, np.nan)
    atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ADX (14)
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 14)  # Need ATR and ADX periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_aligned[i]) or 
            np.isnan(atr_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price deviation from VWAP in ATR units
        deviation = (close[i] - vwap_aligned[i]) / atr_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for mean reversion entries in strong trend
            # Long: price below VWAP by >1.0*ATR AND strong trend
            if (deviation < -1.0 and 
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price above VWAP by >1.0*ATR AND strong trend
            elif (deviation > 1.0 and 
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend weakens
            if (deviation >= 0.0 or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to VWAP or trend weakens
            if (deviation <= 0.0 or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_VWAP_ATR_ADX_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0