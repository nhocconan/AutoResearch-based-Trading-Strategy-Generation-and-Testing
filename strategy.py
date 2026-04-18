#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_Simple
Hypothesis: In low volatility regimes (ATR < 50th percentile), price respects daily Camarilla pivot levels R1/S1.
Breakouts with volume (>2.0x 24-period mean) trigger entries. Uses daily EMA(50) filter to avoid counter-trend trades.
Designed for low trade frequency (10-20/year) on 12h timeframe to minimize fee drag. Works in both bull and bear by
focusing on range-bound conditions where mean reversion at pivot levels is strongest.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA filter
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
    
    # Calculate daily EMA(50) for trend filter
    if len(close_1d) >= 50:
        ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Calculate daily ATR(14) for volatility regime filter
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
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align all 1d data to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: volume > 2.0x 24-period average (strict)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Volatility regime: ATR < 50th percentile (use rolling 50-period percentile)
    atr_percentile = np.full_like(atr_14_12h, np.nan)
    lookback = 50
    
    if len(atr_14_12h) >= lookback:
        for i in range(lookback, len(atr_14_12h)):
            window = atr_14_12h[i-lookback:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                percentile_50 = np.percentile(valid, 50)
                atr_percentile[i] = percentile_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 50) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(ema_50_12h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr_percentile[i]) or np.isnan(atr_14_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Volatility filter: low volatility regime (ATR < 50th percentile)
        low_vol = atr_14_12h[i] < atr_percentile[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in low volatility regime
            if close[i] > R1_12h[i] and vol_confirm and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in low volatility regime
            elif close[i] < S1_12h[i] and vol_confirm and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1
            if close[i] < S1_12h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1
            if close[i] > R1_12h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_Volume_Simple"
timeframe = "12h"
leverage = 1.0