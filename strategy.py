#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with weekly trend filter and volume confirmation
# Long when price breaks above weekly bearish fractal resistance AND weekly close > weekly EMA34 (uptrend) AND volume > 2.0 * 20-bar avg volume
# Short when price breaks below weekly bullish fractal support AND weekly close < weekly EMA34 (downtrend) AND volume > 2.0 * 20-bar avg volume
# Exit when price retraces to the weekly pivot point (average of weekly OHLC)
# Williams fractals require 2-bar confirmation delay on weekly timeframe
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly EMA34 provides strong trend filter for better regime adaptation in both bull and bear markets
# Volume threshold set to 2.0x to reduce false breakouts while maintaining sufficient trade frequency

name = "6h_WilliamsFractal_1wEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for fractal and trend filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams fractals on weekly data (requires 2-bar confirmation delay)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Initialize fractal arrays
    bearish_fractal = np.full(len(high_1w), np.nan)  # resistance fractal (high point)
    bullish_fractal = np.full(len(high_1w), np.nan)  # support fractal (low point)
    
    # Williams fractal: middle bar is highest/lowest of 5 bars
    # Bearish fractal: high[i] is highest of [i-2, i-1, i, i+1, i+2]
    # Bullish fractal: low[i] is lowest of [i-2, i-1, i, i+1, i+2]
    for i in range(2, len(high_1w) - 2):
        # Bearish fractal: current high is highest of surrounding 5 bars
        if (high_1w[i] >= high_1w[i-2] and high_1w[i] >= high_1w[i-1] and 
            high_1w[i] >= high_1w[i+1] and high_1w[i] >= high_1w[i+2]):
            bearish_fractal[i] = high_1w[i]
        
        # Bullish fractal: current low is lowest of surrounding 5 bars
        if (low_1w[i] <= low_1w[i-2] and low_1w[i] <= low_1w[i-1] and 
            low_1w[i] <= low_1w[i+1] and low_1w[i] <= low_1w[i+2]):
            bullish_fractal[i] = low_1w[i]
    
    # Align HTF indicators to 6h timeframe with proper delays
    # Weekly EMA34 needs only the completed weekly bar delay (handled by align_htf_to_ltf)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams fractals need 2 extra weekly bars for confirmation (formation + 2 bars after)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal, additional_delay_bars=2)
    
    # Calculate weekly pivot point (average of weekly OHLC) for exit
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (open_1w + high_1w + low_1w + close_1w) / 4.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Williams Fractal breakout signals with trend and volume filters
            # Long: Break above bearish fractal resistance AND uptrend AND volume spike
            if close[i] > bearish_fractal_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below bullish fractal support AND downtrend AND volume spike
            elif close[i] < bullish_fractal_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price retraces to weekly pivot point (mean reversion)
            if close[i] <= weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price retraces to weekly pivot point (mean reversion)
            if close[i] >= weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals