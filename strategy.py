#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_EMA_Trend_Filter
Hypothesis: 6h Elder Ray (Bull/Bear Power) with zero-lag EMA trend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price above zero-lag EMA(50), and volume > 1.5x MA.
Short when Bull Power < 0, Bear Power > 0, price below zero-lag EMA(50), and volume > 1.5x MA.
Uses 1d trend filter (price vs 1d EMA200) to avoid counter-trend trades in bear markets.
Zero-lag EMA reduces lag while maintaining smoothness. Discrete sizing (0.25) minimizes fee churn.
Target: 12-37 trades/year on 6h. Works in bull/bear by following 1d trend with 6h timing.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    # Trend: price above/below EMA200
    bullish_trend_1d = close_1d > ema200_1d
    bearish_trend_1d = close_1d < ema200_1d
    
    # Align 1d trend to 6h timeframe
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend_1d.astype(float))
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend_1d.astype(float))
    
    # Calculate zero-lag EMA(50) on 6h close
    # Zero-lag EMA = 2*EMA - EMA(EMA) to reduce lag
    ema1 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema2 = pd.Series(ema1).ewm(span=50, adjust=False, min_periods=50).mean().values
    zl_ema = 2 * ema1 - ema2
    
    # Calculate Elder Ray components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ZLEMA, 20 for vol MA, 13 for EMA13)
    start_idx = max(50, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(zl_ema[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(bullish_trend_aligned[i]) or 
            np.isnan(bearish_trend_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price above ZLEMA, bullish 1d trend, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > zl_ema[i] and bullish_trend_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price below ZLEMA, bearish 1d trend, volume spike
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  close[i] < zl_ema[i] and bearish_trend_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Bull Power <= 0 OR Bear Power >= 0 OR price below ZLEMA OR bearish 1d trend
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                close[i] < zl_ema[i] or not bullish_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Bull Power >= 0 OR Bear Power <= 0 OR price above ZLEMA OR bullish 1d trend
            if (bull_power[i] >= 0 or bear_power[i] <= 0 or 
                close[i] > zl_ema[i] or not bearish_trend_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_ZeroLag_EMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0