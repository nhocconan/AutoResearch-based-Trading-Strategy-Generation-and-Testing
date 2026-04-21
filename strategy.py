#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion_V1
Hypothesis: 6h Williams VixFix (WVF) identifies volatility spikes and mean reversion opportunities in 6h timeframe. 
Long when WVF > 0.8 (extreme fear) and price < BB lower band; Short when WVF > 0.8 and price > BB upper band.
Uses 1d HTF trend filter (price > EMA50 for long bias, < EMA50 for short bias) to avoid fighting the trend.
ATR-based stoploss via signal=0 when price moves against position by 2.5*ATR.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
Works in both bull/bear markets: WVF captures panic spikes regardless of regime, HTF filter provides directional bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams VixFix (WVF) - measures market fear
    # WVF = ((Highest Close in Lookback - Low) / (Highest Close in Lookback)) * 100
    lookback = 22
    highest_close = pd.Series(close_6h).rolling(window=lookback, min_periods=lookback).max().values
    wvf = ((highest_close - low_6h) / highest_close) * 100
    wvf = wvf / 100  # normalize to 0-1 range
    
    # Bollinger Bands (20, 2.0) for mean reversion levels
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_6h).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = sma + (bb_std * std)
    bb_lower = sma - (bb_std * std)
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(wvf[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) 
            or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: Extreme fear (WVF > 0.8) + price at/below lower BB + long bias from HTF
            if wvf[i] > 0.8 and price <= bb_lower[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Extreme fear (WVF > 0.8) + price at/above upper BB + short bias from HTF
            elif wvf[i] > 0.8 and price >= bb_upper[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back above middle BB or fear subsides
            elif price >= sma[i] or wvf[i] < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price back below middle BB or fear subsides
            elif price <= sma[i] or wvf[i] < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion_V1"
timeframe = "6h"
leverage = 1.0