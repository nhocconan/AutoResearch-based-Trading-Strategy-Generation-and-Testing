#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day ATR filter and volume confirmation.
# Long when price breaks above 20-period Donchian high, ATR(14) > 1.5x ATR(50), volume > 1.5x 20-period average.
# Short when price breaks below 20-period Donchian low, ATR(14) > 1.5x ATR(50), volume > 1.5x 20-period average.
# Exit when price crosses the 10-period moving average in the opposite direction.
# Uses Donchian channels for breakout signals, ATR for volatility filter, volume for confirmation.
# Works in trending markets (both bull and bear) by capturing breakouts with volatility and volume filters.
# Target: 15-25 trades/year per symbol.
name = "12h_Donchian20_ATR_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for ATR(50) calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ma_10[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high, volatility filter, volume confirmation
            if (close[i] > donchian_high[i] and 
                atr_14[i] > 1.5 * atr_50[i] and 
                volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, volatility filter, volume confirmation
            elif (close[i] < donchian_low[i] and 
                  atr_14[i] > 1.5 * atr_50[i] and 
                  volume[i] > 1.5 * vol_ma_20[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-period MA
            if close[i] < ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-period MA
            if close[i] > ma_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals