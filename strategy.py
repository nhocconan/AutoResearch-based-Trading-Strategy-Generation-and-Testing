#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h VWAP trend + volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + price > 12h VWAP + volume > 2x average
# Short when price breaks below 4h Donchian lower (20-period) + price < 12h VWAP + volume > 2x average
# Exit when price crosses 12h VWAP in opposite direction or opposite Donchian breakout
# Uses 12h VWAP for trend filter, Donchian for entry/exit, volume for confirmation
# Target: 50-150 trades over 4 years with controlled frequency

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate VWAP for each 12h bar
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_array = vwap_12h.values
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h_array)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(50, n):
        # Check for NaN values
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vwap_12h_aligned[i])):
            continue
        
        # Volume confirmation (> 2x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:  # No position - look for entries
            if volume_confirm:
                # Long: price breaks above Donchian upper + price > 12h VWAP
                if close[i] > donchian_upper[i] and close[i] > vwap_12h_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below Donchian lower + price < 12h VWAP
                elif close[i] < donchian_lower[i] and close[i] < vwap_12h_aligned[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:  # Long position - exit on VWAP cross below or opposite breakout
            if close[i] < vwap_12h_aligned[i] or close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit on VWAP cross above or opposite breakout
            if close[i] > vwap_12h_aligned[i] or close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hVWAP_Volume"
timeframe = "4h"
leverage = 1.0