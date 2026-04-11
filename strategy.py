#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Camarilla pivot levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels from previous day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # H2 = close + 1.0 * (high - low)
    # L2 = close - 1.0 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    daily_range = high_1d - low_1d
    
    h4_1d = close_1d + 1.5 * daily_range
    l4_1d = close_1d - 1.5 * daily_range
    h3_1d = close_1d + 1.25 * daily_range
    l3_1d = close_1d - 1.25 * daily_range
    h2_1d = close_1d + 1.0 * daily_range
    l2_1d = close_1d - 1.0 * daily_range
    
    # Shift by 1 to use only completed daily bars
    h4_1d = np.roll(h4_1d, 1)
    l4_1d = np.roll(l4_1d, 1)
    h3_1d = np.roll(h3_1d, 1)
    l3_1d = np.roll(l3_1d, 1)
    h2_1d = np.roll(h2_1d, 1)
    l2_1d = np.roll(l2_1d, 1)
    
    h4_1d[0] = np.nan
    l4_1d[0] = np.nan
    h3_1d[0] = np.nan
    l3_1d[0] = np.nan
    h2_1d[0] = np.nan
    l2_1d[0] = np.nan
    
    # Align daily Camarilla levels to 4h timeframe
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h2_1d_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    l2_1d_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h2_1d_aligned[i]) or np.isnan(l2_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Long breakout: price breaks above H3 with volume
        long_signal = volume_confirmed and (price_close > h3_1d_aligned[i])
        
        # Short breakdown: price breaks below L3 with volume
        short_signal = volume_confirmed and (price_close < l3_1d_aligned[i])
        
        # Exit conditions: price returns to H2 (long exit) or L2 (short exit)
        exit_long = position == 1 and price_close < h2_1d_aligned[i]
        exit_short = position == -1 and price_close > l2_1d_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: 4h Camarilla breakout with volume confirmation.
# Uses daily Camarilla pivot levels (H3/L3 for breakout, H2/L2 for exit) to identify
# institutional support/resistance. Enters long when 4h price breaks above daily H3
# with volume confirmation (>1.5x average volume). Enters short when price breaks
# below daily L3 with volume confirmation. Exits when price returns to daily H2/L2.
# Works in both bull and bear markets by trading breakouts in the direction of
# institutional levels. Volume confirmation ensures participation from market actors.
# Target: 20-50 trades per year to minimize fee drag on 4h timeframe. Camarilla levels
# are widely watched by institutions, providing high-probability breakout levels.