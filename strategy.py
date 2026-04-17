#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band Squeeze Breakout + 1d ADX Regime Filter.
Long when price breaks above upper BB with 1d ADX > 25 (trending) or ADX < 20 with mean-reversion bounce from lower BB.
Short when price breaks below lower BB with 1d ADX > 25 or ADX < 20 with rejection at upper BB.
Uses Bollinger Bands (20,2) on 6h for volatility squeeze/expansion and 1d ADX for regime.
Targets low-volatility breakouts in trending markets and mean-reversion bounces in ranging markets.
Designed to work in both bull (trend breaks) and bear (mean-reversion in range) markets.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) with proper Wilder smoothing
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
            minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Initial ATR
        atr = np.zeros_like(tr)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Avoid division by zero
        atr_safe = np.where(atr == 0, 1e-10, atr)
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr_safe)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr_safe)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 6h Bollinger Bands (20,2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_stddev * bb_std)
    lower_bb = sma - (bb_stddev * bb_std)
    bb_width = (upper_bb - lower_bb) / sma  # Normalized width for squeeze detection
    
    # Calculate 1d ADX and align
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for BB and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        adx_val = adx_14_aligned[i]
        is_squeeze = bb_width[i] < 0.02  # Low volatility threshold
        is_expansion = bb_width[i] > 0.03  # Volatility expansion threshold
        
        # Regime determination
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long conditions:
            # 1. Trending breakout: price crosses above upper BB during expansion
            # 2. Ranging mean reversion: price bounces off lower BB with rejection of lower band
            breakout_long = is_trending and is_expansion and price > upper_bb[i] and close[i-1] <= upper_bb[i-1]
            bounce_long = is_ranging and price > lower_bb[i] and close[i-1] <= lower_bb[i] and low[i] < lower_bb[i]
            
            if breakout_long or bounce_long:
                signals[i] = 0.25
                position = 1
            
            # Short conditions:
            # 1. Trending breakout: price crosses below lower BB during expansion
            # 2. Ranging mean reversion: price rejects upper BB with failure to break above
            breakout_short = is_trending and is_expansion and price < lower_bb[i] and close[i-1] >= lower_bb[i-1]
            bounce_short = is_ranging and price < upper_bb[i] and close[i-1] >= upper_bb[i] and high[i] > upper_bb[i]
            
            if breakout_short or bounce_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB or opposite signal emerges
            if price < sma[i] or (is_ranging and price > upper_bb[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB or opposite signal emerges
            if price > sma[i] or (is_ranging and price < lower_bb[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BBSqueeze_1dADX_Regime"
timeframe = "6h"
leverage = 1.0