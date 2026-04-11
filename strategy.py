#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_pivot_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return signals
    
    # Calculate 1d close for Camarilla pivot (previous day's close)
    close_1d = df_1d['close'].values
    # Camarilla pivot levels based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first value to NaN since no previous day exists
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels for previous day
    range_1d = prev_high - prev_low
    camarilla_h4 = prev_close + 1.5 * range_1d  # Resistance 4
    camarilla_l4 = prev_close - 1.5 * range_1d  # Support 4
    camarilla_h3 = prev_close + 1.25 * range_1d # Resistance 3
    camarilla_l3 = prev_close - 1.25 * range_1d # Support 3
    camarilla_h2 = prev_close + 1.0 * range_1d  # Resistance 2
    camarilla_l2 = prev_close - 1.0 * range_1d  # Support 2
    camarilla_h1 = prev_close + 0.5 * range_1d  # Resistance 1
    camarilla_l1 = prev_close - 0.5 * range_1d  # Support 1
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: 20-period average on 4h
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume_current > 1.5 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Price breaks above Camarilla H4 with volume confirmation
        price_above_h4 = price_close > camarilla_h4_aligned[i]
        if price_above_h4 and vol_confirm:
            enter_long = True
        
        # Short: Price breaks below Camarilla L4 with volume confirmation
        price_below_l4 = price_close < camarilla_l4_aligned[i]
        if price_below_l4 and vol_confirm:
            enter_short = True
        
        # Exit conditions: price returns to Camarilla H3/L3 (mean reversion)
        exit_long = price_close < camarilla_h3_aligned[i]
        exit_short = price_close > camarilla_l3_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout on daily timeframe with volume confirmation.
# Uses previous day's Camarilla levels (H4/L4 for breakout entries, H3/L3 for mean reversion exits).
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Works in both bull and breakout scenarios by capturing volatility expansion breakouts.
# Reduced position size to 0.25 to manage risk. Target: 20-40 trades/year to minimize fee drag.