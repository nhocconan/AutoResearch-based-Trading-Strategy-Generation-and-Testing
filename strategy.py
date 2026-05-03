#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA34 trend filter and volume confirmation
# Williams Fractals identify potential reversal points. We trade breakouts from the most recent
# confirmed fractal in the direction of the 1d trend, with volume confirmation to filter noise.
# Works in bull markets via upside breaks above bullish fractals in uptrends and in bear markets
# via downside breaks below bearish fractals in downtrends. Designed for 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsFractal_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Fractals on 1d data (requires 5 bars: 2 left, center, 2 right)
    # Bearish fractal: high[n] is highest among [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: low[n] is lowest among [n-2, n-1, n, n+1, n+2]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align fractals to 6h timeframe with 2-bar extra delay for confirmation
    # (fractals need 2 subsequent 1d bars to confirm)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Volume confirmation: 20-period EMA on 6h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above the most recent bullish fractal in uptrend with volume spike
            if (not np.isnan(bullish_fractal_aligned[i]) and 
                close[i] > bullish_fractal_aligned[i] and 
                ema_34_1d_aligned[i] < close[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below the most recent bearish fractal in downtrend with volume spike
            elif (not np.isnan(bearish_fractal_aligned[i]) and 
                  close[i] < bearish_fractal_aligned[i] and 
                  ema_34_1d_aligned[i] > close[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below the most recent bullish fractal or loses uptrend alignment
            if (not np.isnan(bullish_fractal_aligned[i]) and close[i] < bullish_fractal_aligned[i]) or \
               ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above the most recent bearish fractal or loses downtrend alignment
            if (not np.isnan(bearish_fractal_aligned[i]) and close[i] > bearish_fractal_aligned[i]) or \
               ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals