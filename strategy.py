#!/usr/bin/env python3
"""
6h_WilliamsFractal_Donchian_Breakout_12hTrend_VolumeSpike
Hypothesis: Trade 6h Donchian(20) breakouts in the direction of 12h trend, confirmed by daily Williams Fractal (breakout structure) and volume spike. Williams Fractal adds structural confirmation of swing high/low validity, reducing false breakouts. Donchian provides objective breakout levels. Volume spike ensures participation. Designed to work in both bull and bear markets by trading with the 12h trend and requiring multiple confirmations. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for Williams Fractal (needs extra delay for confirmation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractal calculation
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Williams Fractal on 1d (requires 2 extra bars for confirmation)
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Add 2-bar delay for fractal confirmation (needs 2 future 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # Calculate Donchian(20) on 6h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(34), Donchian(20), volume MA
    start_idx = max(34, 20, 20) + 2  # +2 for fractal alignment delay
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(bearish_fractal_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        trend_12h_up = close_val > ema_34_12h_aligned[i]   # 12h uptrend
        trend_12h_down = close_val < ema_34_12h_aligned[i]  # 12h downtrend
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND 12h trend up AND bullish fractal AND volume spike
            long_signal = (close_val > highest_high[i]) and trend_12h_up and bullish_fractal_aligned[i] and vol_spike
            
            # Short: price breaks below Donchian low AND 12h trend down AND bearish fractal AND volume spike
            short_signal = (close_val < lowest_low[i]) and trend_12h_down and bearish_fractal_aligned[i] and vol_spike
            
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
            # Exit: 12h trend flips down OR price retracement to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (not trend_12h_up) or (close_val < midpoint):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: 12h trend flips up OR price retracement to midpoint
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            if (not trend_12h_down) or (close_val > midpoint):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsFractal_Donchian_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0