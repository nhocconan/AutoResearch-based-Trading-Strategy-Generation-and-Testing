#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elders Force Index with 1d RSI trend filter and volume confirmation.
# Long when Force Index crosses above zero and daily RSI > 50 (bullish trend).
# Short when Force Index crosses below zero and daily RSI < 50 (bearish trend).
# Uses volume > 1.5x 20-period average for confirmation.
# Force Index = (Close - Close_prev) * Volume
# Daily RSI filter avoids counter-trend trades.
# Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_elders_force_1d_rsi_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Force Index: (Close - Previous Close) * Volume
    close_series = pd.Series(close)
    force_index = (close_series.diff() * volume)
    force_index = force_index.values
    
    # Daily RSI (14-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    daily_bullish = rsi_values > 50  # Bullish when RSI > 50
    daily_bearish = rsi_values < 50   # Bearish when RSI < 50
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if daily RSI data not available
        if np.isnan(daily_bullish_aligned[i]) or np.isnan(daily_bearish_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits
        if position == 1:  # long position
            # Exit: Force Index crosses below zero or daily turn bearish
            if (force_index[i] < 0 and force_index[i-1] > 0) or daily_bearish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Force Index crosses above zero or daily turn bullish
            if (force_index[i] > 0 and force_index[i-1] < 0) or daily_bullish_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and daily trend filter
            if volume_filter:
                # Long: Force Index crosses above zero during bullish day
                if (force_index[i] > 0 and force_index[i-1] <= 0) and daily_bullish_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Force Index crosses below zero during bearish day
                elif (force_index[i] < 0 and force_index[i-1] >= 0) and daily_bearish_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals