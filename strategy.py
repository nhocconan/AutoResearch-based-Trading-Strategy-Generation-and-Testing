#!/usr/bin/env python3
"""
6h_MidPrice_MeanReversion_WeeklyTrend_Filter
Strategy: Mean reversion to 6h mid-price (HLC/3) with weekly trend filter.
Long: Price < mid-price by 0.5*ATR + weekly close > weekly SMA(50) + volume > 1.5x avg
Short: Price > mid-price by 0.5*ATR + weekly close < weekly SMA(50) + volume > 1.5x avg
Exit: Price crosses back to mid-price or ATR-based stop
Position size: 0.25
Designed for mean reversion in ranging markets with trend filter to avoid fighting strong trends.
Works in both bull and bear by using weekly trend filter to align with higher timeframe direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h mid-price (HLC/3)
    mid_price = (high + low + close) / 3.0
    
    # Calculate ATR(14) for volatility normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly SMA(50) for trend filter
    sma_50_weekly = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma_50_weekly)
    
    # Calculate volume average (20-period)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(mid_price[i]) or np.isnan(atr[i]) or 
            np.isnan(sma_50_weekly_aligned[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Weekly trend filter
        weekly_uptrend = weekly_close[-1] > sma_50_weekly[-1] if len(weekly_close) > 0 else False
        weekly_downtrend = weekly_close[-1] < sma_50_weekly[-1] if len(weekly_close) > 0 else False
        
        # Distance from mid-price in ATR units
        dist_from_mid = (close[i] - mid_price[i]) / atr[i] if atr[i] > 0 else 0
        
        if position == 0:
            # Long: Price below mid-price + weekly uptrend + volume filter
            if dist_from_mid < -0.5 and weekly_uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price above mid-price + weekly downtrend + volume filter
            elif dist_from_mid > 0.5 and weekly_downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses back to mid-price
            if dist_from_mid >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses back to mid-price
            if dist_from_mid <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MidPrice_MeanReversion_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0