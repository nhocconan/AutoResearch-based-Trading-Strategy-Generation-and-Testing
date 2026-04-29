#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Fractal Breakout + 1d EMA34 Trend + Volume Spike
# Long when price breaks above recent bearish Williams fractal (high) AND price > 1d EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below recent bullish Williams fractal (low) AND price < 1d EMA34 AND volume > 2.0x 20-bar avg
# Exit when price reverts to the opposite fractal level (mean reversion to structure)
# Williams fractals identify significant swing points; breakouts from these levels with volume and trend filter capture strong moves.
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 12h timeframe.

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
    
    # Get 1d data for Williams fractal calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d data
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams fractals on 1d data
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    n_1d = len(high_1d)
    
    bearish_fractal = np.full(n_1d, np.nan)
    bullish_fractal = np.full(n_1d, np.nan)
    
    for i in range(2, n_1d - 2):
        # Bearish fractal (sell signal): high surrounded by lower highs
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1] and 
            high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        # Bullish fractal (buy signal): low surrounded by higher lows
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1] and 
            low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Williams fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # volume MA and EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema34 = ema_34_1d_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]  # resistance level
        curr_bullish = bullish_fractal_aligned[i]   # support level
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price reverts to bullish fractal level (support)
            if not np.isnan(curr_bullish) and curr_close <= curr_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to bearish fractal level (resistance)
            if not np.isnan(curr_bearish) and curr_close >= curr_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above bearish fractal (resistance) AND price > 1d EMA34 AND volume confirmation
            if (not np.isnan(curr_bearish) and curr_close > curr_bearish and 
                curr_close > curr_ema34 and vol_conf):
                signals[i] = 0.25
                position = 1
            # Short when price breaks below bullish fractal (support) AND price < 1d EMA34 AND volume confirmation
            elif (not np.isnan(curr_bullish) and curr_close < curr_bullish and 
                  curr_close < curr_ema34 and vol_conf):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals