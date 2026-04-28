#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams Fractal breakout with 6h EMA20 trend filter and volume confirmation.
# Enter long when price breaks above the most recent bearish Williams fractal (swing high) with volume > 2.0x average and close > 6h EMA20 (bullish bias).
# Enter short when price breaks below the most recent bullish Williams fractal (swing low) with volume > 2.0x average and close < 6h EMA20 (bearish bias).
# Exit when price crosses the 6h EMA20 in the opposite direction.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-150 total trades over 4 years.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Williams Fractals provide structure from completed 1d swings (no look-ahead), EMA20 filters for trend alignment, volume confirms conviction.

name = "6h_WilliamsFractal_Breakout_6hEMA20_VolumeConfirm_v1"
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
    
    # Get 1d data for Williams Fractals (MTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal (swing high): high[n-2] > high[n-3] and high[n-2] > high[n-1] and high[n-2] > high[n-4] and high[n-2] > high[n]
    # Bullish fractal (swing low): low[n-2] < low[n-3] and low[n-2] < low[n-1] and low[n-2] < low[n-4] and low[n-2] < low[n]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Align Williams Fractals to 6h timeframe with 2-bar extra delay for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 6h data for EMA20 trend filter
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h EMA20
    close_6h = df_6h['close'].values
    ema_20_6h = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_20_6h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 6h EMA20 bias
        bullish_bias = close[i] > ema_20_6h_aligned[i]
        bearish_bias = close[i] < ema_20_6h_aligned[i]
        
        # Fractal breakout conditions
        long_breakout = close[i] > bearish_fractal_aligned[i]
        short_breakout = close[i] < bullish_fractal_aligned[i]
        
        # Exit conditions: cross EMA20 in opposite direction
        long_exit = close[i] < ema_20_6h_aligned[i]
        short_exit = close[i] > ema_20_6h_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
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