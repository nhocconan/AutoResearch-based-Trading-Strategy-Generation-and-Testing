#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above 12h Donchian upper channel with price above 12h EMA20 and volume > 1.5x average.
# Short when price breaks below 12h Donchian lower channel with price below 12h EMA20 and volume > 1.5x average.
# Exit when price returns to 12h EMA20 or opposite Donchian breakout occurs.
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for Donchian channels and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    donch_len = 20
    upper_12h = pd.Series(high_12h).rolling(window=donch_len, min_periods=donch_len).max().values
    lower_12h = pd.Series(low_12h).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Calculate 12h EMA20 for trend filter
    ema_len = 20
    ema_12h = pd.Series(close_12h).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    
    # Align indicators to 4h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_len, ema_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_12h_aligned[i]) or 
            np.isnan(lower_12h_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian upper AND above 12h EMA20
            if (close[i] > upper_12h_aligned[i] and 
                close[i] > ema_12h_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below 12h Donchian lower AND below 12h EMA20
            elif (close[i] < lower_12h_aligned[i] and 
                  close[i] < ema_12h_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to 12h EMA20 or breaks below 12h Donchian lower
            if (close[i] <= ema_12h_aligned[i] or 
                close[i] < lower_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to 12h EMA20 or breaks above 12h Donchian upper
            if (close[i] >= ema_12h_aligned[i] or 
                close[i] > upper_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12hDonchian_EMA20_Volume_v1"
timeframe = "4h"
leverage = 1.0