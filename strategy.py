#!/usr/bin/env python3
"""
6h_1w_1d_IBS_Trend_v1
Hypothesis: Intraday Bar Strength (IBS) combined with 1d trend and weekly structure produces high-probability mean-reversion entries in 6h timeframe.
Long: IBS < 0.3 (oversold) + 1d close > 1d EMA50 (uptrend) + price above weekly S1 (support intact)
Short: IBS > 0.7 (overbought) + 1d close < 1d EMA50 (downtrend) + price below weekly R1 (resistance intact)
Uses volume confirmation and ATR-based stops. Works in bull/bear markets by fading extremes within the trend.
Target: 12-30 trades/year per symbol (50-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # Align weekly levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 24-period average (4 days on 6h)
    vol_ma = prices['volume'].rolling(window=24, min_periods=24).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # IBS (Intraday Bar Strength) = (close - low) / (high - low)
    ibs = (close - low) / (high - low)
    ibs[(high - low) == 0] = 0.5  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ibs[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.3 * vol_ma[i]
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: oversold (IBS < 0.3) in uptrend with volume, price above weekly S1
            if uptrend and volume_ok and ibs[i] < 0.3 and price > s1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought (IBS > 0.7) in downtrend with volume, price below weekly R1
            elif downtrend and volume_ok and ibs[i] > 0.7 and price < r1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: IBS > 0.7 (overbought) or stoploss
            if ibs[i] > 0.7 or price < s1_aligned[i] - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: IBS < 0.3 (oversold) or stoploss
            if ibs[i] < 0.3 or price > r1_aligned[i] + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_IBS_Trend_v1"
timeframe = "6h"
leverage = 1.0