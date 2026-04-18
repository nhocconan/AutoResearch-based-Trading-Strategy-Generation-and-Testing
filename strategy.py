#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h session-based breakout with volume confirmation and volatility regime filter.
# Uses London/NY session overlap (08:00-12:00 UTC) for higher probability breakouts.
# Enters on 12h breakouts above/below 20-period high/low with volume > 2x average.
# Filters by volatility regime using ATR ratio (ATR10/ATR30 < 0.8 = low vol = better breakout).
# Designed for 15-25 trades/year to minimize fee drag while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: London/NY overlap (08:00-12:00 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours < 12)
    
    # Calculate 12h ATR(10) and ATR(30) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    atr_30 = pd.Series(tr).ewm(alpha=1/30, adjust=False, min_periods=30).mean().values
    
    # Volatility regime: low volatility environment (ATR10/ATR30 < 0.8)
    vol_regime = atr_10 < (0.8 * atr_30)
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need ATR30, high/low20, volMA20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_10[i]) or np.isnan(atr_30[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Combined filters: session + volatility regime + volume
        combined_filter = session_filter[i] and vol_regime[i] and vol_filter[i]
        
        if position == 0:
            # Long entry: break above 20-period high with filters
            if high[i] > high_20[i-1] and combined_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: break below 20-period low with filters
            elif low[i] < low_20[i-1] and combined_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: break below 20-period low or end of session
            if low[i] < low_20[i-1] or not session_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above 20-period high or end of session
            if high[i] > high_20[i-1] or not session_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_SessionBreakout_Vol_VolRegime"
timeframe = "12h"
leverage = 1.0