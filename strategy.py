#!/usr/bin/env python3
"""
6h_1d_MarketProfile_ValueArea_Breakout
Hypothesis: On 6B timeframe, trade breakouts of the previous day's value area (VAH/VAL) with volume confirmation.
Value area represents 70% of volume traded in the prior session (TPO-based approximation using volume-weighted close).
Breaks of VAH/VAL indicate institutional participation and tend to trend. Works in bull/bear by only taking breaks
in the direction of the prior day's close relative to the value area (bullish if close > VAH, bearish if close < VAL).
Targets 15-30 trades/year by requiring volume > 1.3x average and clear value area break.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_MarketProfile_ValueArea_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for value area calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Approximate value area using volume-weighted price bins (TPO simulation)
    # We'll use a simplified version: value area = range where 70% of volume occurred
    vah = np.full(len(df_1d), np.nan)  # Value Area High
    val = np.full(len(df_1d), np.nan)  # Value Area Low
    
    for i in range(len(df_1d)):
        if i < 1:  # Need at least one full day
            continue
        # Get the day's data
        day_high = df_1d['high'].iloc[i]
        day_low = df_1d['low'].iloc[i]
        day_close = df_1d['close'].iloc[i]
        day_volume = df_1d['volume'].iloc[i]
        
        # Create price bins between low and high
        num_bins = 30
        bin_edges = np.linspace(day_low, day_high, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Approximate volume distribution: assume volume spreads across the range
        # Weight more toward close price (realistic for crypto)
        weights = np.exp(-0.5 * ((bin_centers - day_close) / ((day_high - day_low) / 4))**2)
        weights = weights / weights.sum() * day_volume
        
        # Find value area (70% of volume)
        sorted_idx = np.argsort(weights)[::-1]
        cum_vol = np.cumsum(weights[sorted_idx])
        va_indices = sorted_idx[cum_vol <= 0.7 * day_volume]
        
        if len(va_indices) > 0:
            vah[i] = bin_centers[va_indices].max()
            val[i] = bin_centers[va_indices].min()
        else:
            vah[i] = day_close
            val[i] = day_close
    
    # Shift to use previous day's value area (no look-ahead)
    vah = np.roll(vah, 1)
    val = np.roll(val, 1)
    vah[0] = np.nan
    val[0] = np.nan
    
    # Align to 6h timeframe
    vah_6h = align_htf_to_ltf(prices, df_1d, vah)
    val_6h = align_htf_to_ltf(prices, df_1d, val)
    
    # Volume average (20 period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(vah_6h[i]) or np.isnan(val_6h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike: current volume > 1.3x average
        volume_spike = volume[i] > vol_ma[i] * 1.3
        
        # Value area break conditions
        vah_break = close[i] > vah_6h[i]
        val_break = close[i] < val_6h[i]
        
        # Entry conditions: break of value area with volume
        long_entry = vah_break and volume_spike
        short_entry = val_break and volume_spike
        
        # Exit conditions: return to opposite value area edge
        long_exit = close[i] < val_6h[i]  # Return to VAL
        short_exit = close[i] > vah_6h[i]  # Return to VAH
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals