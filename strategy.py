#!/usr/bin/env python3
"""
4h_1d_Combined_Momentum_and_Volume_Surge
Hypothesis: Combine price momentum with volume surge and volatility contraction.
Long when: Price above 20-period SMA, RSI above 50, volume > 1.8x 20-period average, and Bollinger Band width contracting.
Short when: Price below 20-period SMA, RSI below 50, volume > 1.8x 20-period average, and Bollinger Band width contracting.
Exit when momentum fades (RSI crosses 50) or volatility expands.
Uses Bollinger Band width as a regime filter to trade during low volatility periods before breakouts.
Aims for 20-40 trades per year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Combined_Momentum_and_Volume_Surge"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands and width on close
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean()
    bb_std = close_series.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    bb_width_avg = bb_width.rolling(window=20, min_periods=20).mean()
    bb_width_values = bb_width.values
    bb_width_avg_values = bb_width_avg.values
    
    # Calculate RSI
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume moving average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Price relative to 20-period SMA
    sma_20 = bb_middle.values  # Same as BB middle
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any data invalid
        if (np.isnan(sma_20[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bb_width_values[i]) or 
            np.isnan(bb_width_avg_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Conditions
        price_above_sma = close[i] > sma_20[i]
        price_below_sma = close[i] < sma_20[i]
        rsi_bullish = rsi_values[i] > 50
        rsi_bearish = rsi_values[i] < 50
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        volatility_contracting = bb_width_values[i] < bb_width_avg_values[i]
        
        # Entry conditions
        long_entry = price_above_sma and rsi_bullish and volume_surge and volatility_contracting
        short_entry = price_below_sma and rsi_bearish and volume_surge and volatility_contracting
        
        # Exit conditions: momentum fade or volatility expansion
        long_exit = not (price_above_sma and rsi_bullish) or bb_width_values[i] > bb_width_avg_values[i] * 1.2
        short_exit = not (price_below_sma and rsi_bearish) or bb_width_values[i] > bb_width_avg_values[i] * 1.2
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals