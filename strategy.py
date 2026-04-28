#!/usr/bin/env python3
"""
6h_LiquidityVoid_Trap_Reversal
Hypothesis: Price often reverses after creating liquidity voids (fair value gaps) on 6h timeframe.
In ranging markets, price revisits these voids before continuing. In trending markets,
voids act as support/resistance. We enter when price returns to fill a void with
confluence from 1d trend (price vs 50 EMA) and volume confirmation.
Designed for low trade frequency (~15-25/year) to minimize fee decay in choppy markets.
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
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Detect 6h fair value gaps (liquidity voids)
    # Bullish FVG: gap between low[i-2] and high[i] where low[i-2] > high[i]
    # Bearish FVG: gap between high[i-2] and low[i] where high[i-2] < low[i]
    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Bullish FVG: two candles ago low > current candle high (gap up)
        if low[i-2] > high[i]:
            fvg_bull[i] = True
        # Bearish FVG: two candles ago high < current candle low (gap down)
        if high[i-2] < low[i]:
            fvg_bear[i] = True
    
    # Track active FVG zones (price hasn't filled them yet)
    # For bullish FVG: active while price hasn't reached the gap low
    # For bearish FVG: active while price hasn't reached the gap high
    active_bull_fvg = np.zeros(n, dtype=bool)
    active_bear_fvg = np.zeros(n, dtype=bool)
    
    # Track the gap boundaries
    bull_fvg_low = np.full(n, np.nan)
    bull_fvg_high = np.full(n, np.nan)
    bear_fvg_low = np.full(n, np.nan)
    bear_fvg_high = np.full(n, np.nan)
    
    for i in range(2, n):
        if fvg_bull[i]:
            bull_fvg_low[i] = high[i]      # Bottom of gap
            bull_fvg_high[i] = low[i-2]    # Top of gap
            active_bull_fvg[i] = True
        elif i > 0:
            bull_fvg_low[i] = bull_fvg_low[i-1]
            bull_fvg_high[i] = bull_fvg_high[i-1]
            active_bull_fvg[i] = active_bull_fvg[i-1]
            # If price touches or goes below gap bottom, gap is filled
            if low[i] <= bull_fvg_high[i]:
                active_bull_fvg[i] = False
        
        if fvg_bear[i]:
            bear_fvg_low[i] = high[i-2]    # Bottom of gap
            bear_fvg_high[i] = low[i]      # Top of gap
            active_bear_fvg[i] = True
        elif i > 0:
            bear_fvg_low[i] = bear_fvg_low[i-1]
            bear_fvg_high[i] = bear_fvg_high[i-1]
            active_bear_fvg[i] = active_bear_fvg[i-1]
            # If price touches or goes above gap top, gap is filled
            if high[i] >= bear_fvg_low[i]:
                active_bear_fvg[i] = False
    
    # Align 1d trend data to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    d1_uptrend = close > ema_50_aligned
    d1_downtrend = close < ema_50_aligned
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_surge[i]) or
            np.isnan(bull_fvg_low[i]) or np.isnan(bear_fvg_high[i])):
            signals[i] = 0.0
            continue
        
        # Long setup: price fills bullish FVG (returns to gap) with trend and volume
        long_setup = (active_bull_fvg[i] and 
                     low[i] <= bull_fvg_high[i] and  # Price touched gap top
                     high[i] >= bull_fvg_low[i] and  # Price entered gap
                     d1_uptrend[i] and 
                     volume_surge[i])
        
        # Short setup: price fills bearish FVG (returns to gap) with trend and volume
        short_setup = (active_bear_fvg[i] and
                      high[i] >= bear_fvg_low[i] and   # Price touched gap bottom
                      low[i] <= bear_fvg_high[i] and   # Price entered gap
                      d1_downtrend[i] and
                      volume_surge[i])
        
        # Exit when price moves through the gap (opposite side)
        long_exit = active_bull_fvg[i] and high[i] > bull_fvg_high[i]
        short_exit = active_bear_fvg[i] and low[i] < bear_fvg_low[i]
        
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Close long
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.25   # Close short
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_LiquidityVoid_Trap_Reversal"
timeframe = "6h"
leverage = 1.0