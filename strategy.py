#!/usr/bin/env python3
"""
1d_ema_trend_volume_filter_v1
Hypothesis: On daily timeframe, use EMA(50) trend filter with volume confirmation.
Long when price crosses above EMA(50) with volume > 1.5x 20-day average.
Short when price crosses below EMA(50) with volume > 1.5x 20-day average.
Exit when price crosses back over EMA(50) in opposite direction.
Designed for 10-20 trades/year to minimize fee drag while capturing major trends.
Works in bull markets (captures uptrends) and bear markets (captures downtrends) by following the trend with volume confirmation.
"""

import numpy as np
import pandas as pd

name = "1d_ema_trend_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA(50) with proper min_periods
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses back below EMA(50)
            if close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses back above EMA(50)
            if close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price crosses above EMA(50)
                if close[i] > ema_50[i] and close[i-1] <= ema_50[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price crosses below EMA(50)
                elif close[i] < ema_50[i] and close[i-1] >= ema_50[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals