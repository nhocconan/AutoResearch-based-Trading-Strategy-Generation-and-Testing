#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d Trend Filter and Volume Confirmation
# Long when Bull Power > 0 and Bear Power < 0 (bullish bias) + price > 1d EMA50 + volume spike
# Short when Bear Power > 0 and Bull Power < 0 (bearish bias) + price < 1d EMA50 + volume spike
# Exit when power signals reverse or volume drops
# Works in bull/bear by using 1d trend filter and Elder Ray's bull/bear power dynamics

name = "6h_ElderRay_Power_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-day average volume for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Elder Ray Power (13-period EMA) on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_ema13 = high - ema_13  # Bull Power
    low_ema13 = low - ema_13    # Bear Power
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(high_ema13[i]) or np.isnan(low_ema13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 6h volume > 1.5x 20-period average of aligned 1d volume MA
        vol_filter = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        # Trend filter: price relative to 1d EMA50
        price_above_ema50 = close[i] > ema_50_1d_aligned[i]
        price_below_ema50 = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive, Bear Power negative, price above 1d EMA50, volume spike
            if (high_ema13[i] > 0 and low_ema13[i] < 0 and price_above_ema50 and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive, Bull Power negative, price below 1d EMA50, volume spike
            elif (low_ema13[i] > 0 and high_ema13[i] < 0 and price_below_ema50 and vol_filter):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Power signals reverse or volume drops
            if (high_ema13[i] <= 0 or low_ema13[i] >= 0 or not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Power signals reverse or volume drops
            if (low_ema13[i] <= 0 or high_ema13[i] >= 0 or not vol_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals