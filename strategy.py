# 2025-06-23 | 4h_PriceAction_Confluence_V3
# Hypothesis: High-probability breakout strategy using confluence of price action (candle close outside Bollinger Bands),
# volume confirmation, and multi-timeframe trend alignment (4h EMA50 vs 1d EMA200). Designed for low trade frequency
# (20-40 trades/year) to minimize fee drag while capturing strong trending moves. Works in both bull and bear markets
# by requiring alignment with higher timeframe trend, reducing false breakouts during ranging periods.

#!/usr/bin/env python3
name = "4h_PriceAction_Confluence_V3"
timeframe = "4h"
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
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_up = np.full(n, np.nan)
    bb_dn = np.full(n, np.nan)
    
    if n >= bb_period:
        sma[bb_period-1] = np.mean(close[0:bb_period])
        for i in range(bb_period, n):
            sma[i] = sma[i-1] + (close[i] - close[i-bb_period]) / bb_period
        
        # Calculate variance
        var = np.full(n, np.nan)
        for i in range(bb_period-1, n):
            if i == bb_period-1:
                var[i] = np.mean((close[0:bb_period] - sma[i])**2)
            else:
                var[i] = var[i-1] + ((close[i] - sma[i])**2 - (close[i-bb_period+1] - sma[i-bb_period+1])**2) / bb_period
        
        std_dev = np.sqrt(np.maximum(var, 0))
        bb_up = sma + bb_std * std_dev
        bb_dn = sma - bb_std * std_dev
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (ema_200_1d[i-1] * 199 + close_1d[i]) / 200
    
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    if n >= vol_period:
        vol_ma[vol_period-1] = np.mean(volume[0:vol_period])
        for i in range(vol_period, n):
            vol_ma[i] = (vol_ma[i-1] * (vol_period-1) + volume[i]) / vol_period
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma > 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(bb_period, vol_period, 200)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(bb_up[i]) or np.isnan(bb_dn[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: close above upper BB AND uptrend (close > EMA200) AND volume confirmation
            if (close[i] > bb_up[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Enter short: close below lower BB AND downtrend (close < EMA200) AND volume confirmation
            elif (close[i] < bb_dn[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit long: close below middle Bollinger Band (mean reversion signal)
            if close[i] < sma[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above middle Bollinger Band (mean reversion signal)
            if close[i] > sma[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals