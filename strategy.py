#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based stoploss.
# Enter long when price breaks above 20-period high + volume > 1.5x 20-bar average.
# Enter short when price breaks below 20-period low + volume > 1.5x 20-bar average.
# Exit on ATR(14) trailing stop: long exits when price < highest_high_since_entry - 2.5*ATR,
# short exits when price > lowest_low_since_entry + 2.5*ATR.
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid excessive fee churn.
# Donchian channels provide structural breakout levels; volume confirms breakout strength;
# ATR stoploss adapts to volatility and limits drawdown in ranging/bear markets.

name = "4h_Donchian20_Breakout_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 14)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Use previous bar's high to avoid look-ahead
        breakout_down = close[i] < donchian_low[i-1]  # Use previous bar's low
        
        # ATR-based trailing stoploss
        if position == 1:  # Long position
            highest_since_entry = max(highest_since_entry, high[i])
            exit_level = highest_since_entry - 2.5 * atr[i]
            exit_long = close[i] < exit_level
        elif position == -1:  # Short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            exit_level = lowest_since_entry + 2.5 * atr[i]
            exit_short = close[i] > exit_level
        else:
            exit_long = False
            exit_short = False
        
        # Handle entries and exits
        if breakout_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
            entry_bar = i
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif breakout_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
            entry_bar = i
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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