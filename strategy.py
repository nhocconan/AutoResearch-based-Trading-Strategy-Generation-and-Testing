#!/usr/bin/env python3
"""
6h_WilliamsFractal_DailyBreakout_v1
Hypothesis: On 6h timeframe, trade breakouts confirmed by daily Williams Fractals. 
In bull markets, price breaks above prior daily high with bullish fractal confirmation. 
In bear markets, price breaks below prior daily low with bearish fractal confirmation. 
Volume spike filters for institutional participation. 
Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
Uses 1d Williams Fractals with 2-bar confirmation delay to avoid look-ahead.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals and daily high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractal calculation
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Williams fractals need 2 extra 1d bars after center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Daily high and low for breakout levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of fractal calculation (5), volume MA (20)
    start_idx = max(5, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        bear_fractal = bearish_fractal_aligned[i]
        bull_fractal = bullish_fractal_aligned[i]
        dh_val = daily_high_aligned[i]
        dl_val = daily_low_aligned[i]
        
        if position == 0:
            # Long: break above prior daily high with bullish fractal and volume spike
            long_signal = (high_val > dh_val) and bull_fractal and vol_spike
            
            # Short: break below prior daily low with bearish fractal and volume spike
            short_signal = (low_val < dl_val) and bear_fractal and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below prior daily low OR bullish fractal fails
            if low_val < dl_val or not bull_fractal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above prior daily high OR bearish fractal fails
            if high_val > dh_val or not bear_fractal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_DailyBreakout_v1"
timeframe = "6h"
leverage = 1.0