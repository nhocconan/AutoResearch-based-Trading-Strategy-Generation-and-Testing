#!/usr/bin/env python3
"""
12h_Keltner_Middle_Bounce_1dTrend_Volume
Hypothesis: Price bouncing from Keltner middle line (EMA20) with daily trend filter and volume confirmation.
Keltner channels adapt to volatility; middle line acts as dynamic support/resistance.
In uptrend (price > daily EMA50), buy near middle; in downtrend, sell near middle.
Volume > 1.5x 24-period average confirms bounce strength. Designed for low trade frequency.
Works in both bull (buy dips) and bear (sell rallies) markets.
"""

name = "12h_Keltner_Middle_Bounce_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Keltner Channel middle line: EMA20 of typical price
    typical_price = (high + low + close) / 3
    ema_20 = np.full_like(typical_price, np.nan)
    if len(typical_price) >= 20:
        ema_20[19] = np.mean(typical_price[0:20])
        for i in range(20, len(typical_price)):
            ema_20[i] = (ema_20[i-1] * 19 + typical_price[i]) / 20
    
    # Volume confirmation: current volume / 24-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 20)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price near EMA20 (within 0.5%) AND uptrend AND volume confirmation
            if (abs(close[i] - ema_20[i]) / ema_20[i] < 0.005 and  # within 0.5%
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price near EMA20 (within 0.5%) AND downtrend AND volume confirmation
            elif (abs(close[i] - ema_20[i]) / ema_20[i] < 0.005 and  # within 0.5%
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Exit long: price moves away from EMA20 (>1.5%) OR trend reversal
                if (abs(close[i] - ema_20[i]) / ema_20[i] > 0.015 or  # beyond 1.5%
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Exit short: price moves away from EMA20 (>1.5%) OR trend reversal
                if (abs(close[i] - ema_20[i]) / ema_20[i] > 0.015 or  # beyond 1.5%
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals