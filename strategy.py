#!/usr/bin/env python3
# 2025-06-22 | 1d_WickReversal_MomentumFilter
# Hypothesis: Daily wicks indicate rejection of price extremes. Long when lower wick > 2x upper wick in downtrend,
# short when upper wick > 2x lower wick in uptrend. Weekly trend filter avoids counter-trend trades.
# Volume confirmation ensures conviction. Designed for low trade frequency (10-25/year) to minimize fee drag.
# Works in bull/bear via trend filter and mean-reversion logic at extremes.

name = "1d_WickReversal_MomentumFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA21 for trend filter
    ema_21_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 21:
        ema_21_1w[20] = np.mean(close_1w[0:21])
        for i in range(21, len(close_1w)):
            ema_21_1w[i] = (ema_21_1w[i-1] * 20 + close_1w[i]) / 21
    
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily calculations
    body = np.abs(close - open_) if 'open' in prices.columns else np.abs(close - np.roll(close, 1))
    upper_wick = high - np.maximum(close, open_) if 'open' in prices.columns else high - np.maximum(close, np.roll(close, 1))
    lower_wick = np.minimum(close, open_) - low if 'open' in prices.columns else np.roll(close, 1) - low
    
    # Handle first bar
    if 'open' in prices.columns:
        open_ = prices['open'].values
        body = np.abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
    else:
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        body = np.abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
    
    # Wick ratio: lower/upper or upper/lower
    lower_to_upper = np.divide(lower_wick, upper_wick, out=np.full_like(lower_wick, np.nan), where=upper_wick!=0)
    upper_to_lower = np.divide(upper_wick, lower_wick, out=np.full_like(upper_wick, np.nan), where=lower_wick!=0)
    
    # Volume spike: current vs 20-day average
    vol_ma20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma20[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.divide(volume, vol_ma20, out=np.full_like(volume, np.nan), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 21)  # Volume MA and weekly EMA ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(lower_to_upper[i]) or np.isnan(upper_to_lower[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: strong lower wick rejection in weekly downtrend + volume
            if (lower_to_upper[i] > 2.0 and 
                close[i] < ema_21_1w_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: strong upper wick rejection in weekly uptrend + volume
            elif (upper_to_lower[i] > 2.0 and 
                  close[i] > ema_21_1w_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: wick exhaustion or trend change
            if lower_to_upper[i] < 0.5 or close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: wick exhaustion or trend change
            if upper_to_lower[i] < 0.5 or close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals