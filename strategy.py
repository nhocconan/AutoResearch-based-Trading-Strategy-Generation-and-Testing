#!/usr/bin/env python3
# 4h_WilliamsVIXFix_MeanReversion_1dTrend
# Hypothesis: In mean-reverting markets, Williams VIX Fix identifies oversold/overbought conditions.
# Long when VIX Fix > 80 (extreme fear) and price > daily EMA50 (uptrend filter).
# Short when VIX Fix < 20 (extreme complacency) and price < daily EMA50 (downtrend filter).
# Exit when VIX Fix reverts to 50 (neutral). Uses 1d trend filter to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) and works in both bull and bear markets by fading extremes with trend alignment.

name = "4h_WilliamsVIXFix_MeanReversion_1dTrend"
timeframe = "4h"
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
    
    # Calculate Williams VIX Fix: measures market fear
    # VIX Fix = (Highest Close in lookback - Low) / (Highest Close - Lowest Low) * 100
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    denominator = highest_close - lowest_low
    vix_fix = np.where(denominator != 0, (highest_close - low) / denominator * 100, 50)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: avoid low-volume false signals
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA50 to 4h timeframe
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(vix_fix[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        vix = vix_fix[i]
        daily_trend = daily_ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Avoid trading in extremely low volume (liquidity risk)
        if volume[i] < 0.5 * vol_ma_val:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Extreme fear (VIX Fix > 80) with uptrend bias (price > EMA50)
            if vix > 80 and close[i] > daily_trend:
                signals[i] = 0.25
                position = 1
            # SHORT: Extreme complacency (VIX Fix < 20) with downtrend bias (price < EMA50)
            elif vix < 20 and close[i] < daily_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VIX Fix returns to neutral (50) or fear subsides
            if vix < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VIX Fix returns to neutral (50) or complacency subsides
            if vix > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals