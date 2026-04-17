#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA trend filter and ATR-based volatility regime.
Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA50 (bullish regime) AND ATR(20) < ATR(50) (low volatility).
Short when Bear Power < 0 AND Bull Power > 0 AND price < 1d EMA50 (bearish regime) AND ATR(20) < ATR(50) (low volatility).
Exit when power signals reverse or volatility expands (ATR(20) > ATR(50) * 1.2).
Uses 1d for EMA50 trend filter, 6h for Elder Ray and ATR calculation.
Designed to capture momentum shifts in low-volatility environments, working in both bull (long bias) and bear (short bias) markets.
Target: 12-30 trades/year per symbol.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Elder Ray Index components
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 6h ATR for volatility regime
    # ATR(20) and ATR(50) for volatility comparison
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # first bar
    atr20 = pd.Series(tr1).rolling(window=20, min_periods=20).mean().values
    atr50 = pd.Series(tr1).rolling(window=50, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(atr20[i]) or
            np.isnan(atr50[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        bullish_regime = close[i] > ema50_1d_aligned[i]
        bearish_regime = close[i] < ema50_1d_aligned[i]
        
        # Volatility filter: low volatility environment (ATR contracting)
        low_volatility = atr20[i] < atr50[i]
        high_volatility_exit = atr20[i] > atr50[i] * 1.2
        
        # Elder Ray signals
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, bullish regime, low volatility
            if (bull_power_positive and bear_power_negative and 
                bullish_regime and low_volatility):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, bearish regime, low volatility
            elif (not bull_power_positive and not bear_power_negative and 
                  bearish_regime and low_volatility):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power becomes positive OR volatility expands
            if (not bear_power_negative or high_volatility_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power becomes negative OR volatility expands
            if (bull_power_positive or high_volatility_exit):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA50Trend_ATRVolatility"
timeframe = "6h"
leverage = 1.0