#!/usr/bin/env python3
"""
6H_ELDER_RAY_BULL_POWER_REVERSAL
Hypothesis: Elder Ray Bull Power (high - EMA13) and Bear Power (low - EMA13) with 1d trend filter.
In bull markets (price > 1d EMA50), look for Bear Power exhaustion (low - EMA13 crosses above -threshold) for long entries.
In bear markets (price < 1d EMA50), look for Bull Power exhaustion (high - EMA13 crosses below +threshold) for short entries.
Uses volatility-adjusted thresholds to adapt to market conditions. Designed to capture reversal points in trending markets.
Targets 15-25 trades/year to minimize fee drain with high-probability setups.
Works in both bull (mean reversion within uptrend) and bear (mean reversion within downtrend) markets.
"""

name = "6H_ELDER_RAY_BULL_POWER_REVERSAL"
timeframe = "6h"
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
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power: high - EMA13
    bull_power = high - ema13
    
    # Bear Power: low - EMA13
    bear_power = low - ema13
    
    # ATR for volatility normalization (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema50_1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(atr[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility-adjusted thresholds
        bull_threshold = atr[i] * 0.3  # Bull Power must exceed 0.3*ATR to be strong
        bear_threshold = -atr[i] * 0.3  # Bear Power must be below -0.3*ATR to be strong
        
        if position == 0:
            # Determine market regime based on 1d trend
            uptrend = close[i] > ema50_1d_aligned[i]
            
            if uptrend:
                # In uptrend: look for Bear Power exhaustion (bullish reversal)
                # Enter long when Bear Power crosses above bear_threshold from below
                if bear_power[i] > bear_threshold and bear_power[i-1] <= bear_threshold:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            else:
                # In downtrend: look for Bull Power exhaustion (bearish reversal)
                # Enter short when Bull Power crosses below bull_threshold from above
                if bull_power[i] < bull_threshold and bull_power[i-1] >= bull_threshold:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
        elif position == 1:
            # Exit long when Bull Power shows strength (uptrend resumption) or Bear Power weakens
            if bull_power[i] > bull_threshold or bear_power[i] < bear_threshold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Bear Power shows strength (downtrend resumption) or Bull Power weakens
            if bear_power[i] < bear_threshold or bull_power[i] > bull_threshold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals