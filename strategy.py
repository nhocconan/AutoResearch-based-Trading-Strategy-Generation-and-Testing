#!/usr/bin/env python3
"""
6h_fvg_breakout_1d_trend_volume_v1
Hypothesis: Fair Value Gaps (FVGs) on 6h act as institutional support/resistance. 
Breaking through an FVG with volume and daily trend alignment indicates strong momentum.
In bull markets, buy the breakout of bullish FVG; in bear markets, sell breakdown of bearish FVG.
Targets 15-35 trades/year by requiring confluence of FVG, volume, and trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_fvg_breakout_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Identify Fair Value Gaps (FVG) on 6h
    # Bullish FVG: gap between low[i-2] and high[i] where low[i] > high[i-2]
    # Bearish FVG: gap between high[i-2] and low[i] where high[i] < low[i-2]
    fvg_bull_top = np.roll(high, 2)  # high[i-2]
    fvg_bull_bot = np.roll(low, 2)   # low[i-2]
    fvg_bear_top = np.roll(high, 2)  # high[i-2]
    fvg_bear_bot = np.roll(low, 2)   # low[i-2]
    
    # Bullish FVG exists when low[i] > high[i-2]
    bullish_fvg = low > fvg_bull_top
    # Bearish FVG exists when high[i] < low[i-2]
    bearish_fvg = high < fvg_bear_bot
    
    # Store FVG boundaries
    fvg_bull_low = np.where(bullish_fvg, fvg_bull_bot, np.nan)  # lower bound of bullish FVG
    fvg_bull_high = np.where(bullish_fvg, fvg_bull_top, np.nan) # upper bound of bullish FVG
    fvg_bear_low = np.where(bearish_fvg, fvg_bear_bot, np.nan)  # lower bound of bearish FVG
    fvg_bear_high = np.where(bearish_fvg, fvg_bear_top, np.nan) # upper bound of bearish FVG
    
    # Forward fill FVG levels until they are filled
    # For bullish FVG: act as support until price breaks below
    # For bearish FVG: act as resistance until price breaks above
    fvg_support = pd.Series(fvg_bull_low).ffill().values
    fvg_resistance = pd.Series(fvg_bear_high).ffill().values
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(fvg_support[i]) or 
            np.isnan(fvg_resistance[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below FVG support OR trend turns down
            if close[i] < fvg_support[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above FVG resistance OR trend turns up
            if close[i] > fvg_resistance[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above FVG resistance (bullish FVG broken) + volume + uptrend
            if (close[i] > fvg_resistance[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below FVG support (bearish FVG broken) + volume + downtrend
            elif (close[i] < fvg_support[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals