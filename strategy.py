#!/usr/bin/env python3
"""
1d_1w_Volume_Spike_Camarilla_Pivot_Bounce_v1
Hypothesis: On daily timeframe, price reacts strongly to weekly Camarilla pivot levels (especially L3/H3) when accompanied by volume spikes.
In bull markets: buy dips to L3/H3 with volume confirmation. In bear markets: sell rallies to H3/L3 with volume confirmation.
Uses weekly Camarilla levels calculated from prior week's OHLC, ensuring no look-ahead. Volume spike filter reduces false signals.
Designed for 10-25 trades/year by requiring confluence of pivot level touch + volume spike.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf  # Note: using align_ltf_to_hlf for weekly to daily alignment

name = "1d_1w_Volume_Spike_Camarilla_Pivot_Bounce_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels from prior week's OHLC (no look-ahead)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each week: based on prior week's OHLC
    # H4 = close + 1.5*(high - low)
    # H3 = close + 1.1*(high - low)
    # L3 = close - 1.1*(high - low)
    # L4 = close - 1.5*(high - low)
    # We'll use H3 and L3 as primary levels
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    # Calculate levels based on prior week (shift by 1 to avoid look-ahead)
    wk_high_prev = np.roll(wk_high, 1)
    wk_low_prev = np.roll(wk_low, 1)
    wk_close_prev = np.roll(wk_close, 1)
    # First week has no prior, set to zeros (will be handled by alignment)
    wk_high_prev[0] = 0
    wk_low_prev[0] = 0
    wk_close_prev[0] = 0
    
    wk_range = wk_high_prev - wk_low_prev
    # Camarilla H3 and L3
    camarilla_h3 = wk_close_prev + 1.1 * wk_range
    camarilla_l3 = wk_close_prev - 1.1 * wk_range
    
    # Align weekly levels to daily timeframe (with proper delay for weekly bar close)
    camarilla_h3_daily = align_ltf_to_hlf(prices, df_1w, camarilla_h3)
    camarilla_l3_daily = align_ltf_to_hlf(prices, df_1w, camarilla_l3)
    
    # Volume spike filter: 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_daily[i]) or np.isnan(camarilla_l3_daily[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price proximity to Camarilla levels (within 0.3%)
        dist_to_h3_pct = abs(close[i] - camarilla_h3_daily[i]) / camarilla_h3_daily[i] * 100
        dist_to_l3_pct = abs(close[i] - camarilla_l3_daily[i]) / camarilla_l3_daily[i] * 100
        near_h3 = dist_to_h3_pct <= 0.3
        near_l3 = dist_to_l3_pct <= 0.3
        
        # Volume spike: current volume > 2.0x average
        volume_spike = volume[i] > vol_ma[i] * 2.0
        
        # Entry conditions: bounce off levels with volume
        long_entry = near_l3 and volume_spike  # Bounce off L3 support
        short_entry = near_h3 and volume_spike  # Rejection at H3 resistance
        
        # Exit conditions: price moves back toward midpoint or opposite level
        # Calculate midpoint between H3 and L3 for exit reference
        midpoint = (camarilla_h3_daily[i] + camarilla_l3_daily[i]) / 2
        
        long_exit = close[i] >= midpoint  # Price reached midpoint, take profit
        short_exit = close[i] <= midpoint  # Price reached midpoint, take profit
        
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