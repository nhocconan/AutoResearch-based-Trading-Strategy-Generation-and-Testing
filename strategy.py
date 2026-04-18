#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Price tends to revert to or break from key intraday support/resistance levels
(Camarilla pivots). In trending markets, breaks of R1/S1 levels with volume and
trend alignment (12h EMA) yield sustainable moves. This strategy targets 25-35
trades/year to minimize fee drag while capturing directional moves in both bull
and bear markets. Uses discrete position sizing (0.25) to reduce churn.
"""

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
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    ema34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate prior day's Camarilla levels using prior 6 bars (6*4h = 24h)
    high_prior = pd.Series(high).rolling(window=6, min_periods=6).max().shift(1).values
    low_prior = pd.Series(low).rolling(window=6, min_periods=6).min().shift(1).values
    close_prior = pd.Series(close).rolling(window=6, min_periods=6).last().shift(1).values
    
    # Camarilla levels
    R1 = close_prior + 1.1 * (high_prior - low_prior) / 12
    S1 = close_prior - 1.1 * (high_prior - low_prior) / 12
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ok = vol_filter[i]
        trend = ema34_12h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume, in uptrend
            if price > R1[i] and vol_ok and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume, in downtrend
            elif price < S1[i] and vol_ok and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if price returns to midpoint or trend weakens
            midpoint = (R1[i] + S1[i]) / 2
            if price < midpoint or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if price returns to midpoint or trend weakens
            midpoint = (R1[i] + S1[i]) / 2
            if price > midpoint or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_12hTrend"
timeframe = "4h"
leverage = 1.0