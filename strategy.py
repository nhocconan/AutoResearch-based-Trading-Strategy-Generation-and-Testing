#!/usr/bin/env python3
# 1d_weekly_fractal_breakout_volume_v2
# Hypothesis: Weekly Williams fractal breakout with volume confirmation and daily trend filter.
# Uses weekly bearish/bullish fractal breaks for entry, volume > 1.5x average for confirmation,
# and daily EMA trend filter to avoid counter-trend trades. Works in bull markets by catching
# uptrend continuations and in bear markets by catching downtrend continuations.
# Fractals provide high-probability breakout points, volume filter reduces false signals,
# targeting 20-40 trades/year.

name = "1d_weekly_fractal_breakout_volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for fractals
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate Williams fractals on weekly data
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_weekly, low_weekly)
    # Fractals need 2 extra weekly bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_weekly, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_weekly, bullish_fractal, additional_delay_bars=2)
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.zeros_like(close_daily)
    ema_daily[ema_period-1] = np.mean(close_daily[:ema_period])
    for i in range(ema_period, len(close_daily)):
        ema_daily[i] = (close_daily[i] * 2 + ema_daily[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align daily EMA to daily timeframe (no shift needed as it's same timeframe)
    ema_daily_aligned = ema_daily  # Already on daily timeframe
    
    # Volume filter: 20-period average volume
    vol_ma = np.zeros_like(volume)
    vol_ma[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma[:19] = vol_ma[19]  # Fill beginning with first valid value
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 50 + 5  # EMA period + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if trend reverses or volume fails
            if not uptrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if trend reverses or volume fails
            if not downtrend_htf or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly bullish fractal, volume confirmation, and daily uptrend
            if (close[i] > bullish_fractal_aligned[i] and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly bearish fractal, volume confirmation, and daily downtrend
            elif (close[i] < bearish_fractal_aligned[i] and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals