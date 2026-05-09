#!/usr/bin/env python3
"""
6h_AdaptiveBreakout_Volume_Trend
Hypothesis: Breakouts from 6-hour price channels with volume confirmation and trend filter from 1d EMA.
Uses adaptive channel width based on recent volatility to avoid false breakouts in low volatility periods.
Designed for 6h timeframe to capture multi-day moves while minimizing trade frequency (target: 15-30 trades/year).
Works in both bull and bear markets by requiring trend alignment and volume confirmation.
"""

name = "6h_AdaptiveBreakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h ATR for adaptive channel width (period=14)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = np.full_like(close, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[0:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate dynamic channel: ATR-based bands around SMA(20)
    sma_period = 20
    sma = np.full_like(close, np.nan)
    if len(close) >= sma_period:
        sma[sma_period-1] = np.mean(close[0:sma_period])
        for i in range(sma_period, len(close)):
            sma[i] = (sma[i-1] * (sma_period-1) + close[i]) / sma_period
    
    # Upper and lower bands: SMA ± (ATR * multiplier)
    atr_multiplier = 2.0
    upper_band = sma + atr * atr_multiplier
    lower_band = sma - atr * atr_multiplier
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_ma_period:
        vol_ma[vol_ma_period-1] = np.mean(volume[0:vol_ma_period])
        for i in range(vol_ma_period, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * (vol_ma_period-1) + volume[i]) / vol_ma_period
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(sma_period, vol_ma_period, atr_period, 50)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above upper band AND uptrend (price > EMA50) AND volume spike
            if (close[i] > upper_band[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below lower band AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < lower_band[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 4 bars (1 day)
            if bars_since_entry < 4:
                signals[i] = 0.25
            else:
                # Exit long: price breaks below lower band OR trend reversal (price < EMA50)
                if close[i] < lower_band[i] or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Minimum holding period: 4 bars (1 day)
            if bars_since_entry < 4:
                signals[i] = -0.25
            else:
                # Exit short: price breaks above upper band OR trend reversal (price > EMA50)
                if close[i] > upper_band[i] or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals