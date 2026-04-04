#!/usr/bin/env python3
"""
exp_6551_6h_camarilla1d_pivot_vol_v1
Hypothesis: 6h Camarilla pivot levels from 1d timeframe with volume confirmation.
- Fade at R3/S3 levels (mean reversion in range markets)
- Breakout continuation at R4/S4 levels (trend following in strong moves)
- Volume spike (>1.5x 20-period MA) confirms breakout/fade strength
- Works in both bull/bear markets: Camarilla adapts to volatility, volume confirms institutional participation
- Target: 75-150 total trades over 4 years with discrete sizing (0.25) to minimize fee drag
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6551_6h_camarilla1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOL_MA_PERIOD = 20
VOL_THRESHOLD = 1.5      # volume must be 1.5x its 20-period MA
SIGNAL_SIZE = 0.25       # 25% position size

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align to LTF (6h) with shift(1) for completed bars only
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = VOL_MA_PERIOD + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]):
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i] * VOL_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Exit conditions: opposite Camarilla level touch or volume drying up
        if position == 1:  # long position
            # Exit if price touches/falls below S3 (mean reversion target)
            exit_long = close[i] <= s3_aligned[i]
            # Or if volume dries up significantly
            exit_long = exit_long or (not np.isnan(vol_ma[i]) and volume[i] < vol_ma[i] * 0.5)
            if exit_long:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            # Exit if price touches/rises above R3 (mean reversion target)
            exit_short = close[i] >= r3_aligned[i]
            # Or if volume dries up significantly
            exit_short = exit_short or (not np.isnan(vol_ma[i]) and volume[i] < vol_ma[i] * 0.5)
            if exit_short:
                signals[i] = 0.0
                position = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Long breakout: price breaks above R4 with volume (continuation)
            long_breakout = close[i] > r4_aligned[i] and vol_ok
            # Long fade: price rejects at S3 with volume (mean reversion)
            long_fade = close[i] < s3_aligned[i] and close[i] > s4_aligned[i] and vol_ok
            
            # Short breakout: price breaks below S4 with volume (continuation)
            short_breakout = close[i] < s4_aligned[i] and vol_ok
            # Short fade: price rejects at R3 with volume (mean reversion)
            short_fade = close[i] > r3_aligned[i] and close[i] < r4_aligned[i] and vol_ok
            
            if long_breakout or long_fade:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_breakout or short_fade:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals