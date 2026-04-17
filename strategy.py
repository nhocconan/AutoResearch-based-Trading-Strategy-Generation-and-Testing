#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA200 trend filter and ATR-based volatility filter.
Long when Bull Power > 0 AND price > 1d EMA200 (bullish trend) AND ATR(14) < 1.5 * ATR(50) (low volatility).
Short when Bear Power < 0 AND price < 1d EMA200 (bearish trend) AND ATR(14) < 1.5 * ATR(50) (low volatility).
Exit when price crosses the 1d EMA200 in opposite direction.
Elder Ray measures bull/bear power relative to EMA13, providing early trend strength signals. Combined with 1d EMA200
for primary trend direction and ATR volatility filter to avoid choppy markets. Designed for low trade frequency
(12-37/year) to minimize fee drag while capturing strong trends in both bull and bear markets.
"""

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
    
    # Get 1d data for EMA200 trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA13 on 1d for Elder Ray
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Calculate ATR(14) and ATR(50) on 1d for volatility filter
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    true_range_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    
    # Handle first value for true_range (no previous close)
    true_range_1d[0] = high_low_1d[0]
    
    tr_series = pd.Series(true_range_1d)
    atr_14_1d = tr_series.rolling(window=14, min_periods=14).mean().values
    atr_50_1d = tr_series.rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to 6h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_200 = ema_200_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        atr_14 = atr_14_1d_aligned[i]
        atr_50 = atr_50_1d_aligned[i]
        price = close[i]
        
        # Volatility filter: only trade when ATR(14) < 1.5 * ATR(50) (low volatility environment)
        vol_filter = atr_14 < 1.5 * atr_50
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND price > 1d EMA200 (bullish trend) AND low volatility
            if bull_power > 0 and price > ema_200 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND price < 1d EMA200 (bearish trend) AND low volatility
            elif bear_power < 0 and price < ema_200 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA200
            if price < ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA200
            if price > ema_200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA200_VolatilityFilter"
timeframe = "6h"
leverage = 1.0