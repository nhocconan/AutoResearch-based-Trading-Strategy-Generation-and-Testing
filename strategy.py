#!/usr/bin/env python3
"""
1d_1W_Camarilla_R1S1_Breakout_Volume
Hypothesis: Use weekly market structure to identify bull/bear regimes, then trade daily breakouts at Camarilla R1/S1 levels with volume confirmation.
In bull markets (price > weekly EMA20), buy breaks above R1. In bear markets (price < weekly EMA20), sell breaks below S1.
Volume must be > 1.5x average to confirm breakout strength. Uses only daily timeframe with weekly filter.
Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag and avoid overtrading.
Works in bull via breakouts above R1, works in bear via breakdowns below S1.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Previous day's OHLC for Camarilla calculation (using prior completed day)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day: no prior day, so set to zero (will be filtered out by warmup)
    prev_close[0] = 0
    prev_high[0] = 0
    prev_low[0] = 0
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Weekly trend filter: EMA20 on weekly close
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily and weekly data to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # need enough for weekly EMA and daily calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Determine market regime based on weekly EMA20
        bull_market = close[i] > ema20_1w_aligned[i]
        bear_market = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # In bull market: look for longs on breakout above R1
            if bull_market and close[i] > r1_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # In bear market: look for shorts on breakdown below S1
            elif bear_market and close[i] < s1_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below R1 or weekly trend turns bearish
            if close[i] < r1_aligned[i] or not bull_market:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 or weekly trend turns bullish
            if close[i] > s1_aligned[i] or not bear_market:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Camarilla_R1S1_Breakout_Volume"
timeframe = "1d"
leverage = 1.0