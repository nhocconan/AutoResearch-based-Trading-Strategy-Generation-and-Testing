#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Long when price breaks above Donchian upper band (20-period high) AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band (20-period low) AND volume > 1.5x 20-bar avg
# Exit long when price crosses below Donchian middle (10-period average of high/low) OR trailing stop hit
# Exit short when price crosses above Donchian middle OR trailing stop hit
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
# Donchian channels provide clear structure; volume confirms institutional participation.
# Works in bull markets (breakouts continue trends) and bear markets (breakouts catch reversals).

name = "4h_Donchian20_VolumeConfirm_ATRTrail_v1"
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
    
    # Donchian channels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    start_idx = 20  # Donchian/ATR warmup
    
    for i in range(start_idx, n):
        # Skip if volume MA not ready
        if np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        
        # Donchian levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        middle = donchian_middle[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Update highest high for trailing stop
            highest_high = max(highest_high, curr_high)
            
            # Exit conditions: price < middle OR trailing stop hit
            if curr_close < middle or curr_close < (highest_high - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low for trailing stop
            lowest_low = min(lowest_low, curr_low)
            
            # Exit conditions: price > middle OR trailing stop hit
            if curr_close > middle or curr_close > (lowest_low + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
                lowest_low = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND volume confirmation
            if curr_close > upper and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high = curr_high
            # Short when price breaks below lower band AND volume confirmation
            elif curr_close < lower and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low = curr_low
            else:
                signals[i] = 0.0
    
    return signals