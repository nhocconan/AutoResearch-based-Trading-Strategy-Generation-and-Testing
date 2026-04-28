#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractal breakout with volume confirmation and 1w EMA50 trend filter.
# Enter long when price breaks above recent bullish fractal high with volume spike and above 1w EMA50.
# Enter short when price breaks below recent bearish fractal low with volume spike and below 1w EMA50.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "12h_WilliamsFractal_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    
    bullish_fractal = np.full(n_1d, np.nan)  # High of bullish fractal
    bearish_fractal = np.full(n_1d, np.nan)  # Low of bearish fractal
    
    for i in range(2, n_1d - 2):
        # Bullish fractal: middle bar has highest high
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bullish_fractal[i] = high_1d[i]
        
        # Bearish fractal: middle bar has lowest low
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bearish_fractal[i] = low_1d[i]
    
    # Forward fill to get most recent fractal levels
    bullish_fractal = pd.Series(bullish_fractal).ffill().values
    bearish_fractal = pd.Series(bearish_fractal).ffill().values
    
    # Align 1d Fractal levels to 12h timeframe with 2-bar delay for confirmation (fractals need 2 future bars)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume spike: >1.8x 30-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 1.8 * volume_ma_30
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_ma_30[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA50
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Fractal breakout conditions with volume confirmation
        long_breakout = close[i] > bullish_fractal_aligned[i] and volume_spike[i]
        short_breakout = close[i] < bearish_fractal_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite fractal level or trend reversal
        long_exit = close[i] < bearish_fractal_aligned[i] or below_ema
        short_exit = close[i] > bullish_fractal_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_ema and position >= 0:
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