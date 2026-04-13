#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with volume spike filter only
    # Long: price breaks above H3 AND volume > 2.0x 20-period average
    # Short: price breaks below L3 AND volume > 2.0x 20-period average
    # Exit: price touches opposite Camarilla level (H3 for shorts, L3 for longs) or reverses
    # Using 12h timeframe for low frequency (target 12-37/year), Camarilla levels for structure,
    # volume spike to confirm breakout strength, and discrete sizing (0.25) to minimize fees.
    # No ADX filter to avoid over-filtering; rely on volume confirmation for momentum.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Calculate Camarilla levels: H3, L3
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    camarilla_h3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.25 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: >2.0x 20-period average (strict to reduce trades)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_h3 = close[i] > h3_aligned[i]
        breakout_l3 = close[i] < l3_aligned[i]
        
        # Exit conditions: retest opposite level (mean reversion in range)
        retest_l3 = close[i] < l3_aligned[i] and position == 1  # Long exit on L3 retest
        retest_h3 = close[i] > h3_aligned[i] and position == -1  # Short exit on H3 retest
        
        # Entry logic: Camarilla breakout + volume confirmation
        long_entry = breakout_h3 and volume_spike[i]
        short_entry = breakout_l3 and volume_spike[i]
        
        # Exit logic: retest opposite level
        long_exit = retest_l3
        short_exit = retest_h3
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_h3l3_breakout_volume_only_v1"
timeframe = "12h"
leverage = 1.0