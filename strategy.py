#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Williams Fractal for trend direction and 1d Donchian breakout for entry.
# Weekly Williams Fractal identifies potential turning points on higher timeframe.
# 1d Donchian breakout provides entry signals with confirmation from weekly trend.
# Volume confirmation (>1.5x 20-period average) reduces false breakouts.
# ATR-based stop loss manages risk via signal=0 when price moves against position.
# Designed to work in both bull and bear markets by using weekly fractal to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Williams Fractal calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Fractal: need 2 bars on each side for confirmation
    # Bearish fractal: high[n] is highest of [n-2, n-1, n, n+1, n+2]
    # Bullish fractal: low[n] is lowest of [n-2, n-1, n, n+1, n+2]
    n_1w = len(high_1w)
    bearish_fractal = np.zeros(n_1w, dtype=bool)
    bullish_fractal = np.zeros(n_1w, dtype=bool)
    
    for i in range(2, n_1w - 2):
        # Bearish fractal: current high is highest in window
        if (high_1w[i] >= high_1w[i-2] and high_1w[i] >= high_1w[i-1] and
            high_1w[i] >= high_1w[i+1] and high_1w[i] >= high_1w[i+2]):
            bearish_fractal[i] = True
        # Bullish fractal: current low is lowest in window
        if (low_1w[i] <= low_1w[i-2] and low_1w[i] <= low_1w[i-1] and
            low_1w[i] <= low_1w[i+1] and low_1w[i] <= low_1w[i+2]):
            bullish_fractal[i] = True
    
    # Convert to trend indicator: 1 for bullish fractal (uptrend bias), -1 for bearish fractal (downtrend bias), 0 otherwise
    # We need additional delay of 2 bars for confirmation as per Williams Fractal rules
    fw_fractal = np.where(bullish_fractal, 1, np.where(bearish_fractal, -1, 0)).astype(float)
    
    # Align weekly fractal to daily timeframe with additional 2-bar delay for confirmation
    fw_fractal_aligned = align_htf_to_ltf(prices, df_1w, fw_fractal, additional_delay_bars=2)
    
    # Load daily data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 20)  # Need Donchian and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(fw_fractal_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts above 1d Donchian high or below 1d Donchian low
            # Only trade in direction of weekly Williams Fractal (trend filter)
            
            # Long: price breaks above 1d Donchian high AND weekly bullish fractal
            if (close[i] > donchian_high[i] and 
                fw_fractal_aligned[i] == 1 and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d Donchian low AND weekly bearish fractal
            elif (close[i] < donchian_low[i] and 
                  fw_fractal_aligned[i] == -1 and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 1d Donchian low or weekly fractal turns bearish
            if (close[i] <= donchian_low[i] or 
                fw_fractal_aligned[i] == -1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 1d Donchian high or weekly fractal turns bullish
            if (close[i] >= donchian_high[i] or 
                fw_fractal_aligned[i] == 1):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wWilliamsFractal_1dDonchian_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0