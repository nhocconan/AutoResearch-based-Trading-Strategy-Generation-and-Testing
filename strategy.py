#!/usr/bin/env python3
# 4h_fractal_breakout_12h_trend_volume_v1
# Hypothesis: Use Williams Fractal breakout on 4h with 12h trend filter and volume confirmation.
# Williams Fractals identify potential reversal points; breakout above/below fractal with trend and volume
# captures momentum in both bull and bear markets. 12h trend ensures alignment with higher timeframe.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_fractal_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Fractals on 4h (5-bar: 2 left, 2 right)
    # We need the 4h OHLC data for fractal calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 4h data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_4h['high'].values,
        df_4h['low'].values,
    )
    
    # Align fractals to 4h timeline (they are already on 4h bars)
    # Then align to lower timeframe (4h is our base, so no alignment needed for entry logic)
    # But we need to align the 12h trend to 4h
    
    # 12h EMA trend filter (21-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: volume > 1.5x 20-period average (approx 5 days on 4h)
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    if len(volume) >= vol_period:
        vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(5, vol_period) + 5  # fractal needs 5 bars, plus volume MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Calculate current 4h bar index for fractal arrays
        # Each 4h bar = 16 of our 15m bars, but we're on 4h timeframe
        # Since our prices are 4h data, we can use direct indexing
        idx_4h = i  # because we're operating on 4h data directly
        
        # Need to access fractal arrays - they are calculated on df_4h
        # We need to map our loop index to df_4h index
        # Since we're using 4h prices, and df_4h is the 4h data, they should align
        # But to be safe, we calculate the index in df_4h
        # df_4h starts at the same time as prices[0] for 4h timeframe
        if idx_4h >= len(df_4h):
            # We've run out of 4h data (shouldn't happen if aligned properly)
            signals[i] = 0.0
            continue
            
        bullish_fractal_val = bullish_fractal[idx_4h] if idx_4h < len(bullish_fractal) else np.nan
        bearish_fractal_val = bearish_fractal[idx_4h] if idx_4h < len(bearish_fractal) else np.nan
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below bearish fractal or trend fails
            if close[i] < bearish_fractal_val or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above bullish fractal or trend fails
            if close[i] > bullish_fractal_val or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above bullish fractal with uptrend
                if not np.isnan(bullish_fractal_val) and close[i] > bullish_fractal_val and close[i] > ema_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below bearish fractal with downtrend
                elif not np.isnan(bearish_fractal_val) and close[i] < bearish_fractal_val and close[i] < ema_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals