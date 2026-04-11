#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return signals
    
    # Calculate 1d Camarilla pivot levels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values for pivot calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels
    range_ = prev_high - prev_low
    camarilla_h3 = prev_close + range_ * 1.1 / 4
    camarilla_l3 = prev_close - range_ * 1.1 / 4
    camarilla_h4 = prev_close + range_ * 1.1 / 2
    camarilla_l4 = prev_close - range_ * 1.1 / 2
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long conditions: price breaks above H3 with volume
        long_signal = volume_confirmed and (price_high > camarilla_h3_aligned[i])
        
        # Short conditions: price breaks below L3 with volume
        short_signal = volume_confirmed and (price_low < camarilla_l3_aligned[i])
        
        # Exit when price returns to H4/L4 levels
        exit_long = position == 1 and (price_low < camarilla_h4_aligned[i] or price_high > camarilla_h4_aligned[i])
        exit_short = position == -1 and (price_high > camarilla_l4_aligned[i] or price_low < camarilla_l4_aligned[i])
        
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

# Hypothesis: Camarilla pivot breakout with volume confirmation on 12h timeframe.
# Uses daily Camarilla levels (H3/L3 for entry, H4/L4 for exit) to identify
# institutional support/resistance. Enters long when price breaks above H3 with
# volume confirmation (>1.8x average volume), short when breaks below L3.
# Exits when price returns to H4/L4 levels. Works in both bull and bear markets
# by capturing institutional breakouts. Target: 50-150 total trades over 4 years
# (12-37/year) to minimize fee drag on 12h timeframe. Camarilla levels are
# widely watched by institutions, providing reliable breakout signals.
# Volume confirmation ensures participation from market actors, reducing false breakouts.