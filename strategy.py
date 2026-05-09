#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ChoppinessIndex_Regime_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for choppiness index calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ATR for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CI = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    sum_atr14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll_diff = highest_high - lowest_low
    
    # Avoid division by zero
    chop_raw = np.full_like(sum_atr14, np.nan)
    mask = (hh_ll_diff > 0) & (~np.isnan(sum_atr14)) & (~np.isnan(hh_ll_diff))
    chop_raw[mask] = 100 * np.log10(sum_atr14[mask] / hh_ll_diff[mask]) / np.log10(14)
    
    chop_1d = chop_raw
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12-period RSI for mean reversion signals
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    avg_loss = loss.ewm(alpha=1/12, adjust=False, min_periods=12).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop = chop_1d_aligned[i]
        rsi_val = rsi_values[i]
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Chop > 61.8 indicates ranging market (good for mean reversion)
            if chop > 61.8 and vol_ok:
                # RSI oversold -> long
                if rsi_val < 30:
                    signals[i] = 0.25
                    position = 1
                # RSI overbought -> short
                elif rsi_val > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral or chop drops (trending market)
            if rsi_val > 50 or chop < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral or chop drops (trending market)
            if rsi_val < 50 or chop < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals