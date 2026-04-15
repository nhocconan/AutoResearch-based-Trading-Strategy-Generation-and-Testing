#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter
# Uses Donchian(20) channels for breakout direction, volume > 1.5x 20-bar median for confirmation,
# and ATR-based stoploss to limit downside. Works in both bull and bear markets by
# capturing breakouts in trending regimes while avoiding false signals in ranging markets.
# Target: 20-40 trades/year per symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility filtering and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above upper Donchian band + volume confirmation
        if (close[i] > highest_high[i] and volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower Donchian band + volume confirmation
        elif (close[i] < lowest_low[i] and volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: ATR-based stoploss or opposite breakout
        elif i > 0 and signals[i-1] != 0:
            if signals[i-1] == 0.25:  # Long position
                # Stop if price drops below entry - 2*ATR or opposite breakout
                if (close[i] < highest_high[i-1] - 2.0 * atr[i] or 
                    close[i] < lowest_low[i]):
                    signals[i] = 0.0
                else:
                    signals[i] = signals[i-1]
            else:  # Short position
                # Stop if price rises above entry + 2*ATR or opposite breakout
                if (close[i] > lowest_low[i-1] + 2.0 * atr[i] or 
                    close[i] > highest_high[i]):
                    signals[i] = 0.0
                else:
                    signals[i] = signals[i-1]
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0