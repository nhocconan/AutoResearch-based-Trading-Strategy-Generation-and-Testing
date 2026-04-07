#!/usr/bin/env python3
"""
4h_volatility_breakout_1d_trend_v1
Hypothesis: On 4-hour timeframe, breakout from ATR-based volatility channels (upper/lower bands = close ± ATR*multiplier) with 1-day trend filter (close > SMA50 for long, close < SMA50 for short) and volume confirmation (volume > 20-period average). Captures explosive moves in both bull and bear regimes while avoiding whipsaws in sideways markets. Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_1d_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for volatility channels (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility channels: close ± ATR * multiplier
    atr_multiplier = 2.5
    upper_band = close + atr * atr_multiplier
    lower_band = close - atr * atr_multiplier
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 1-day trend filter: SMA50
    close_series = pd.Series(close)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(sma_50[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below lower band OR trend turns bearish
            if close[i] < lower_band[i] or close[i] < sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above upper band OR trend turns bullish
            if close[i] > upper_band[i] or close[i] > sma_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout long: price closes above upper band with bullish trend
                if close[i] > upper_band[i] and close[i] > sma_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below lower band with bearish trend
                elif close[i] < lower_band[i] and close[i] < sma_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals