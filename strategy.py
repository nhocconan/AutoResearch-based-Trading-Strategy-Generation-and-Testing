#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_volume_v1"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return signals
    
    # Calculate weekly volume average (20-period)
    vol_ma_20 = pd.Series(df_1w['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly high and low for range
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's range
    range_1w = high_1w - low_1w
    
    # Previous week's close for Camarilla calculation
    close_1w = df_1w['close'].values
    
    # Camarilla levels (based on previous week's close)
    l4 = close_1w - (range_1w * 1.1000)
    l3 = close_1w - (range_1w * 1.1000 / 2)
    h3 = close_1w + (range_1w * 1.1000 / 2)
    h4 = close_1w + (range_1w * 1.1000)
    
    # Shift by 1 to use only completed weekly bars
    l4 = np.roll(l4, 1)
    l3 = np.roll(l3, 1)
    h3 = np.roll(h3, 1)
    h4 = np.roll(h4, 1)
    l4[0] = np.nan
    l3[0] = np.nan
    h3[0] = np.nan
    h4[0] = np.nan
    
    # Align weekly Camarilla levels to daily timeframe
    l4_aligned = align_htf_to_ltf(prices, df_1w, l4)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1w, h4)
    
    # Align weekly volume average to daily timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Align previous week's close (pivot point) to daily timeframe
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    prev_close_aligned = align_htf_to_ltf(prices, df_1w, prev_close_1w)
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(l4_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(prev_close_aligned[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Volume confirmation: volume > 2.0x 20-period weekly average
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Long: price breaks above H3/H4 with volume
        long_signal = volume_confirmed and (price_high > h3_aligned[i] or price_high > h4_aligned[i])
        
        # Short: price breaks below L3/L4 with volume
        short_signal = volume_confirmed and (price_low < l3_aligned[i] or price_low < l4_aligned[i])
        
        # Exit when price returns to the previous week's close (pivot point)
        exit_long = position == 1 and price_close < prev_close_aligned[i]
        exit_short = position == -1 and price_close > prev_close_aligned[i]
        
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
        elif position == -1 and exit_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Weekly Camarilla pivot breakout strategy on daily timeframe with volume confirmation.
# Uses 1w Camarilla levels (L3, L4, H3, H4) from the previous week's price action.
# Enters long when price breaks above H3 or H4 with volume confirmation (>2x average weekly volume).
# Enters short when price breaks below L3 or L4 with volume confirmation.
# Exits when price returns to the previous week's close (pivot point).
# Weekly timeframe reduces noise and increases signal reliability.
# Volume confirmation (>2x average) reduces false breakouts.
# Designed for low trade frequency (target: 10-30 trades/year) to minimize fee drag on daily timeframe.
# Works in both bull and bear markets by trading breakouts in the direction of momentum.
# Weekly Camarilla levels provide stronger support/resistance levels than daily levels.
# Targets BTC and ETH primarily, with potential applicability to SOL.