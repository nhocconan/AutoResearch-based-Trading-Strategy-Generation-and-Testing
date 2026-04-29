#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
# Long when price breaks above latest bullish Williams fractal AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below latest bearish Williams fractal AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price reverts to 12h 20-bar EMA (mean reversion to intermediate trend)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Williams fractals provide precise swing high/low breakout levels, 1d EMA34 filters counter-trend moves,
# volume confirmation ensures breakout strength. Works in trending markets (breakouts) and ranges (mean reversion to EMA).

name = "12h_WilliamsFractal_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams fractals and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least 5 days for fractals
        return np.zeros(n)
    
    # Calculate Williams fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams fractal: 5-bar pattern where middle bar is highest/lowest
    bullish_fractal = np.full(len(high_1d), np.nan)
    bearish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        # Bullish fractal: middle bar has lowest low
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
        # Bearish fractal: middle bar has highest high
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
    
    # For breakout trading, we need the most recent completed fractal
    # Williams fractals are lagging - need 2 extra bars after center bar for confirmation
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    
    # Get 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Exit condition: 12h 20-bar EMA (mean reversion to intermediate trend)
    close_series = pd.Series(close)
    ema_20_12h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_fractal_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(ema_20_12h[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_bullish = bullish_fractal_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_ema20 = ema_20_12h[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to 12h 20-bar EMA (mean reversion)
            if curr_close <= curr_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to 12h 20-bar EMA (mean reversion)
            if curr_close >= curr_ema20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above bullish fractal AND price > 1d EMA34 AND volume confirmation
            if (not np.isnan(curr_bullish) and curr_close > curr_bullish and 
                curr_close > curr_ema34 and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below bearish fractal AND price < 1d EMA34 AND volume confirmation
            elif (not np.isnan(curr_bearish) and curr_close < curr_bearish and 
                  curr_close < curr_ema34 and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals