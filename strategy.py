#!/usr/bin/env python3
# 12h_1d_donchian_volume_momentum_v1
# Hypothesis: Trade breakouts of daily Donchian channels with 12h momentum confirmation and volume filter.
# Uses 1-day Donchian(20) breakout for direction, 12h RSI(14) for momentum filter, and volume surge for confirmation.
# In bull markets: buys breakouts above 20-day high with momentum.
# In bear markets: sells breakdowns below 20-day low with momentum.
# Volume filter reduces false breakouts. Target: 20-50 trades/year (80-200 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_donchian_volume_momentum_v1"
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
    
    # Daily Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: highest high/lowest low of past 20 days
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already delayed by rolling window)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 12h RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or momentum fades (RSI < 40)
            if close[i] < donchian_low_aligned[i] or rsi_values[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or momentum fades (RSI > 60)
            if close[i] > donchian_high_aligned[i] or rsi_values[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge and bullish momentum (RSI > 50)
            if (close[i] > donchian_high_aligned[i] and vol_surge and 
                rsi_values[i] > 50):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge and bearish momentum (RSI < 50)
            elif (close[i] < donchian_low_aligned[i] and vol_surge and 
                  rsi_values[i] < 50):
                position = -1
                signals[i] = -0.25
    
    return signals