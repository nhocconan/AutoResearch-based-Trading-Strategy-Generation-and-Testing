# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using weekly pivot points (from 1w) for bias and 
daily ATR for volatility filtering. In bull markets, price tends to stay above 
weekly pivot; in bear markets, below. Weekly pivot provides structural support/resistance
that works across regimes. Daily ATR filters out low-volatility chop. 
Target: 15-30 trades/year per symbol with 0.25 position size.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Weekly data for pivot bias ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (standard calculation)
    # P = (H + L + C) / 3
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === Daily data for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily True Range and ATR(14)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h price action for entry timing ===
    # Simple 6h high/low for breakout detection
    high_6h = high
    low_6h = low
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure weekly pivot and daily ATR are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Calculate dynamic ATR threshold (adaptive to volatility regime)
        # Use 50-period average of ATR to normalize
        atr_ma = pd.Series(atr_1d_aligned[:i+1]).rolling(window=50, min_periods=10).mean().iloc[-1] if i >= 50 else np.mean(atr_1d_aligned[warmup:i+1])
        atr_threshold = atr_ma * 0.5  # Only trade when ATR is above 50% of its average
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly pivot OR volatility drops too low
            if (price < weekly_pivot_val) or (atr_1d_val < atr_threshold):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly pivot OR volatility drops too low
            if (price > weekly_pivot_val) or (atr_1d_val < atr_threshold):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session and when volatility is sufficient
            if in_session and (atr_1d_val >= atr_threshold):
                # LONG: Price above weekly pivot (bullish bias)
                if price > weekly_pivot_val:
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price below weekly pivot (bearish bias)
                elif price < weekly_pivot_val:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyPivot_Bias_DailyATRFilter"
timeframe = "6h"
leverage = 1.0