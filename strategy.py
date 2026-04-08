#!/usr/bin/env python3
# 4h_fractal_breakout_1d_trend_volume_v1
# Hypothesis: Use 1-day Williams Fractal breakouts aligned with 4h price action and volume confirmation.
# In bull markets: buy breaks above recent bullish fractal resistance with volume.
# In bear markets: sell breaks below recent bearish fractal support with volume.
# Uses 4h timeframe for entry timing, 1d for fractal structure, and volume filter to avoid false breakouts.
# Target: 20-40 trades/year via fractal breakouts + volume + trend filter.

name = "4h_fractal_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_htf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for fractal calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals on daily data
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values
    )
    
    # Align fractals to 4h timeframe with 2-bar delay for confirmation
    bullish_fractal_aligned = align_ltf_to_htf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    bearish_fractal_aligned = align_ltf_to_htf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    
    # Volume filter: 20-period average volume on 4h
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning
    
    # 4h EMA (20) for trend filter
    ema_period = 20
    ema = np.zeros_like(close)
    ema[ema_period-1] = np.mean(close[:ema_period])
    for i in range(ema_period, len(close)):
        ema[i] = (close[i] * 2 + ema[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(20, 19) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(ema[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below bearish fractal support
            if close[i] < bearish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bullish fractal resistance
            if close[i] > bullish_fractal_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above bullish fractal resistance with volume and uptrend
            if (close[i] > bullish_fractal_aligned[i] and 
                close[i-1] <= bullish_fractal_aligned[i-1] and  # Just broke above
                volume_filter and 
                close[i] > ema[i]):  # Uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below bearish fractal support with volume and downtrend
            elif (close[i] < bearish_fractal_aligned[i] and 
                  close[i-1] >= bearish_fractal_aligned[i-1] and  # Just broke below
                  volume_filter and 
                  close[i] < ema[i]):  # Downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals