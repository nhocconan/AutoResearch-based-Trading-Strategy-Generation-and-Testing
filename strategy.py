#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation.
# Uses 4h Donchian channel (20-period) for trend direction and 1h for entry timing.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull/bear markets via breakout/breakdown of key levels.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# Position size: 0.20 (20% of capital).

name = "1h_donchian_breakout_4h_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period) for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian upper/lower bands (20-period high/low)
    donch_high_4h = np.full(len(high_4h), np.nan)
    donch_low_4h = np.full(len(low_4h), np.nan)
    
    for i in range(19, len(high_4h)):
        donch_high_4h[i] = np.max(high_4h[i-19:i+1])
        donch_low_4h[i] = np.min(low_4h[i-19:i+1])
    
    # Align 4h Donchian bands to 1h timeframe (shifted by 1 bar for completed bars)
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if Donchian data not available
        if np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches 4h Donchian lower band or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (close[i] <= donch_low_4h_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price reaches 4h Donchian upper band or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (close[i] >= donch_high_4h_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Breakout above 4h Donchian upper band (long)
                if (close[i] > donch_high_4h_aligned[i] and close[i-1] <= donch_high_4h_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Breakdown below 4h Donchian lower band (short)
                elif (close[i] < donch_low_4h_aligned[i] and close[i-1] >= donch_low_4h_aligned[i]):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals