#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Fractal breakouts with 1w EMA50 trend filter and volume confirmation.
# Enter long when price breaks above the most recent bullish fractal (high) and close > 1w EMA50 and volume > 2x 20-bar average.
# Enter short when price breaks below the most recent bearish fractal (low) and close < 1w EMA50 and volume > 2x 20-bar average.
# Exit on opposite fractal break or when price crosses 1w EMA50.
# Williams Fractals identify key swing points where price has shown reversal tendency.
# Using 1w EMA50 ensures alignment with the major weekly trend, reducing counter-trend trades in bear markets.
# Volume confirmation (2x average) adds conviction to breakouts, filtering weak moves.
# Discrete position sizing (0.25) controls risk and minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsFractal_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals (swing points)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bullish fractal: high[i] is the highest among high[i-2], high[i-1], high[i], high[i+1], high[i+2]
    # Bearish fractal: low[i] is the lowest among low[i-2], low[i-1], low[i], low[i+1], low[i+2]
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] >= high_1d[i-2] and high_1d[i] >= high_1d[i-1] and 
            high_1d[i] >= high_1d[i+1] and high_1d[i] >= high_1d[i+2]):
            bullish_fractal[i] = high_1d[i]
        if (low_1d[i] <= low_1d[i-2] and low_1d[i] <= low_1d[i-1] and 
            low_1d[i] <= low_1d[i+1] and low_1d[i] <= low_1d[i+2]):
            bearish_fractal[i] = low_1d[i]
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    # Williams Fractals need extra delay: wait for 2 additional 1d bars after the center bar for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1w EMA50 bias
        bullish_bias = close[i] > ema_50_1w_aligned[i]
        bearish_bias = close[i] < ema_50_1w_aligned[i]
        
        # Current fractal levels (most recent completed fractal)
        bullish_level = bullish_fractal_aligned[i]
        bearish_level = bearish_fractal_aligned[i]
        
        # Entry conditions: price breaks fractal level with volume and trend alignment
        long_entry = (not np.isnan(bullish_level)) and (close[i] > bullish_level) and bullish_bias and vol_confirm
        short_entry = (not np.isnan(bearish_level)) and (close[i] < bearish_level) and bearish_bias and vol_confirm
        
        # Exit conditions: opposite fractal break or EMA50 cross
        long_exit = (not np.isnan(bearish_level) and close[i] < bearish_level) or (close[i] < ema_50_1w_aligned[i])
        short_exit = (not np.isnan(bullish_level) and close[i] > bullish_level) or (close[i] > ema_50_1w_aligned[i])
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals