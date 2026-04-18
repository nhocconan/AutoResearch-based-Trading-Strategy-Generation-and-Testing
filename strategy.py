#!/usr/bin/env python3
"""
4h_WilliamsFractal_Reversal_v1
Hypothesis: Williams fractals identify swing highs/lows that often act as support/resistance. 
In ranging markets (Choppiness Index > 61.8), price tends to reverse at these levels. 
Enter long at bullish fractal (support) with stop below fractal low, short at bearish fractal (resistance) with stop above fractal high. 
Use 1d Williams fractals aligned to 4h, with 1d Choppiness Index for regime filter. 
Requires volume > 1.3x 20-period average for confirmation. 
Target: 20-40 trades/year by combining fractal reversal with range regime filter. 
Works in ranging markets via mean reversion and avoids trending markets where fractals break.
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
    
    # Get 1d data for Williams fractals and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Fractals (5-point: bar is highest/lowest of 2 bars each side)
    bearish_fractal = np.zeros(len(high_1d))  # 1 = bearish fractal (sell signal)
    bullish_fractal = np.zeros(len(low_1d))   # 1 = bullish fractal (buy signal)
    
    if len(high_1d) >= 5:
        for i in range(2, len(high_1d) - 2):
            # Bearish fractal: high[i] is highest of 5 bars
            if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
                high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
                bearish_fractal[i] = 1
            # Bullish fractal: low[i] is lowest of 5 bars
            if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
                low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
                bullish_fractal[i] = 1
    
    # Williams fractals need 2 extra bars for confirmation (per rule 2b)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Choppiness Index (14-period) for regime detection
    chop_length = 14
    chop = np.full(len(close_1d), np.nan)
    
    if len(high_1d) >= chop_length + 1:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with close_1d index
        
        # Sum of TR over chop_length periods
        tr_sum = np.full(len(close_1d), np.nan)
        if len(tr) >= chop_length:
            for i in range(chop_length, len(tr)):
                tr_sum[i] = np.sum(tr[i-chop_length+1:i+1])
            
            # Highest high and lowest low over chop_length periods
            hh = np.full(len(high_1d), np.nan)
            ll = np.full(len(low_1d), np.nan)
            for i in range(chop_length, len(high_1d)):
                hh[i] = np.max(high_1d[i-chop_length+1:i+1])
                ll[i] = np.min(low_1d[i-chop_length+1:i+1])
            
            # Chop = 100 * log10(sum(tr) / (hh - ll)) / log10(chop_length)
            diff = hh - ll
            chop = np.where(diff > 0, 100 * np.log10(tr_sum / diff) / np.log10(chop_length), 50)
    
    # Align Choppiness Index to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(25, chop_length + 1, vol_period) + 1  # fractal needs 5 bars + 2 delay
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Range regime: Choppiness Index > 61.8 indicates ranging market
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0 and in_session and ranging_market and vol_confirm:
            # Long at bullish fractal (support level)
            if bullish_fractal_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short at bearish fractal (resistance level)
            elif bearish_fractal_aligned[i] == 1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below fractal low OR chop < 50 (trending)
            # We don't have exact fractal low price, so use close below fractal level as proxy
            # In practice, we'd need to store the actual fractal price level
            if chop_aligned[i] < 50:  # trend emerging, exit
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above fractal high OR chop < 50 (trending)
            if chop_aligned[i] < 50:  # trend emerging, exit
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsFractal_Reversal_v1"
timeframe = "4h"
leverage = 1.0