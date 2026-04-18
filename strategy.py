#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Filtered_V3
Hypothesis: In low volatility regimes (weekly ATR < 50-day ATR median), price respects daily Camarilla pivot levels R1/S1.
Breakouts with volume (>2.0x 24-period mean) trigger entries. Uses weekly trend filter to avoid counter-trend trades.
Optimized for very low trade frequency (<20/year) on 12h timeframe to minimize fee drag. Works in both bull and bear by
adapting to volatility regime. Uses much stricter volume and volatility filters to reduce overtrading.
"""

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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R1[i] = prev_close + 1.1 * range_ / 12
            S1[i] = prev_close - 1.1 * range_ / 12
    
    # Get weekly data for volatility and trend filters
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14)
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1w = calculate_atr(high_1w, low_1w, close_1w, 14)
    
    # Weekly ATR 50-period median for volatility regime
    atr_median = np.full_like(atr_1w, np.nan)
    if len(atr_1w) >= 50:
        for i in range(49, len(atr_1w)):
            atr_median[i] = np.nanmedian(atr_1w[i-49:i+1])
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all 1w data to 12h timeframe
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_median_12h = align_htf_to_ltf(prices, df_1w, atr_median)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Align daily data to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 2.0x 24-period average (much stricter)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(atr_1w_12h[i]) or np.isnan(atr_median_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: weekly ATR < 50-day median (low volatility regime)
        low_vol_regime = atr_1w_12h[i] < atr_median_12h[i]
        
        # Volume confirmation: much stricter
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend filter: price above weekly EMA (bullish bias)
        bullish_bias = close[i] > ema_1w_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in low volatility regime
            if close[i] > R1_12h[i] and vol_confirm and low_vol_regime and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in low volatility regime (counter-trend only in strong bear)
            elif close[i] < S1_12h[i] and vol_confirm and low_vol_regime and not bullish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 OR volatility increases (regime change)
            if close[i] < S1_12h[i] or atr_1w_12h[i] > atr_median_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 OR volatility increases (regime change)
            if close[i] > R1_12h[i] or atr_1w_12h[i] > atr_median_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Filtered_V3"
timeframe = "12h"
leverage = 1.0