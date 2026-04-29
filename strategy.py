#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg
# Exit when price retraces to Donchian midpoint (mean reversion) or opposite breakout occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year on 4h.
# Donchian channels provide dynamic support/resistance that adapts to volatility.
# Volume confirmation ensures breakouts have conviction, reducing false signals.
# Works in both bull (breakouts continue trend) and bear (breakouts reverse trend) markets.

name = "4h_Donchian20_VolumeBreakout_MidpointExit_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume MA need 20 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        midpoint = donchian_mid[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND volume confirmation
            if curr_high > upper and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND volume confirmation
            elif curr_low < lower and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price retraces to midpoint or breaks below Donchian low
            if curr_close <= midpoint or curr_low < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price retraces to midpoint or breaks above Donchian high
            if curr_close >= midpoint or curr_high > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals