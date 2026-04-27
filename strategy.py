#!/usr/bin/env python3
"""
Hypothesis: 4-hour ATR breakout with daily volume confirmation and weekly trend filter.
Enters long when price breaks above ATR-based upper band with above-average volume and weekly uptrend.
Enters short when price breaks below ATR-based lower band with above-average volume and weekly downtrend.
Uses volatility-based entry to capture breakouts while filtering low-probability setups.
Target: 20-30 trades/year per symbol to minimize fee drag and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR-based bands (like Donchian but volatility-adjusted)
    upper_band = np.roll(close, 1) + 1.5 * atr  # Previous close + 1.5*ATR
    lower_band = np.roll(close, 1) - 1.5 * atr  # Previous close - 1.5*ATR
    
    # Calculate daily volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate weekly close for trend filter
    close_1w = df_1w['close'].values
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR(14) + bands + volume MA
    start_idx = max(14, 20)  # ATR needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(close_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        trend_1w = close_1w_aligned[i]
        
        # Volume filter: volume > 1.2x daily average
        vol_filter = vol_now > 1.2 * vol_ma
        
        # Entry conditions: ATR band breakout with volume + weekly trend
        if position == 0:
            # Long: price breaks above upper band with volume + weekly uptrend
            if price_now > upper_band[i] and vol_filter and price_now > trend_1w:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + weekly downtrend
            elif price_now < lower_band[i] and vol_filter and price_now < trend_1w:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below weekly trend or ATR-based stop
            if price_now < trend_1w or price_now < (close[i-1] - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above weekly trend or ATR-based stop
            if price_now > trend_1w or price_now > (close[i-1] + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ATRBreakout_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0