#!/usr/bin/env python3
"""
1d_Pivot_R1_S1_Breakout_Volume_1wTrend
Hypothesis: Trade Camarilla pivot breakouts on 1-day chart with volume confirmation and weekly trend filter. 
Long when price breaks above R1 with volume > 1.5x 20-day average and weekly close above weekly open (bullish weekly candle). 
Short when price breaks below S1 with volume > 1.5x 20-day average and weekly close below weekly open (bearish weekly candle).
Uses weekly trend to avoid counter-trend trades in strong trends. Targets 15-25 trades/year via strict pivot breakout + volume + trend confluence.
Works in bull markets by catching breakouts, works in bear markets by avoiding longs in downtrends and taking shorts on breakdowns.
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
    
    # Get 1d data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift to get previous day's values
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day has no previous day
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate pivot levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align pivot levels to 1d timeframe (no change, but for consistency)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: bullish if weekly close > weekly open
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    
    # Align weekly trend to 1d timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)  # Need volume MA and at least 1 day of data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + weekly bullish
            if close[i] > R1_aligned[i] and vol_confirm and weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + weekly bearish
            elif close[i] < S1_aligned[i] and vol_confirm and weekly_bullish_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or weekly turns bearish
            if close[i] < S1_aligned[i] or weekly_bullish_aligned[i] < 0.5:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or weekly turns bullish
            if close[i] > R1_aligned[i] or weekly_bullish_aligned[i] > 0.5:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_Volume_1wTrend"
timeframe = "1d"
leverage = 1.0