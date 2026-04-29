#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian(20) high, close > 1d EMA50, and volume > 1.5x 20-period average volume
# Short when price breaks below Donchian(20) low, close < 1d EMA50, and volume > 1.5x 20-period average volume
# Exit when price touches opposite Donchian(20) level or volume drops below average
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Donchian channels provide clear structure, 1d EMA50 filters for higher-timeframe trend alignment,
# volume confirmation ensures breakouts are supported by participation.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.
# Works in both bull and bear markets by only trading with the 1d trend and requiring volume confirmation.

name = "4h_Donchian20_1dEMA50_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian(20) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_vol = volume[i]
        curr_dc_high = high_roll[i]
        curr_dc_low = low_roll[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_vol_ma = vol_ma[i]
        curr_vol_threshold = vol_threshold[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price touches Donchian lower channel OR volume drops below average
            if curr_low <= curr_dc_low or curr_vol < curr_vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches Donchian upper channel OR volume drops below average
            if curr_high >= curr_dc_high or curr_vol < curr_vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian upper channel, close > 1d EMA50, and volume > threshold
            if curr_high > curr_dc_high and curr_close > curr_ema50_1d and curr_vol > curr_vol_threshold:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian lower channel, close < 1d EMA50, and volume > threshold
            elif curr_low < curr_dc_low and curr_close < curr_ema50_1d and curr_vol > curr_vol_threshold:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals