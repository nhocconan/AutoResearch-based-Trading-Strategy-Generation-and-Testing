#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted VWAP with 12h trend filter
# Uses 12h EMA50 for trend direction and 4h VWAP as dynamic support/resistance.
# Price crossing above/below VWAP with volume confirmation signals trend continuation.
# Trend filter ensures alignment with higher timeframe momentum.
# Designed for 4h timeframe to capture multi-day swings with moderate frequency.
# Target: 20-30 trades/year per symbol (80-120 total) to balance opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate VWAP on 4h data
    typical_price = (high + low + close) / 3
    vwap_numerator = (typical_price * volume).cumsum()
    vwap_denominator = volume.cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vwap[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above VWAP with volume confirmation and price > 12h EMA50
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and
                volume[i] > 1.5 * volume[i-1] and close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below VWAP with volume confirmation and price < 12h EMA50
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and
                  volume[i] > 1.5 * volume[i-1] and close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back through VWAP
            if position == 1:
                if close[i] < vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > vwap[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Volume_Cross_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0