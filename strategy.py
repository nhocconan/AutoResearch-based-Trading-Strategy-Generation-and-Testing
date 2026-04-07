#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Momentum + 4h Trend + Volume Filter
# Hypothesis: Combines 1h momentum (price > SMA20) with 4h trend filter (price > SMA50)
# and volume confirmation to capture trend continuations while avoiding whipsaws.
# Works in bull markets via momentum + uptrend, in bear via short momentum + downtrend.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.

name = "1h_momentum_4h_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h SMA50 for trend filter
    sma_50_4h = pd.Series(df_4h['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    
    # 1h SMA20 for momentum filter
    sma_20_1h = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_ok = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(sma_50_4h_aligned[i]) or np.isnan(sma_20_1h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check filters
        vol_filter = vol_ok[i]
        session_filter = session_ok[i]
        
        if position == 1:  # Long position
            # Exit: momentum fails or trend turns bearish
            if close[i] <= sma_20_1h[i] or close[i] <= sma_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: momentum fails or trend turns bullish
            if close[i] >= sma_20_1h[i] or close[i] >= sma_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            if vol_filter and session_filter:
                # Long: momentum up + uptrend
                if close[i] > sma_20_1h[i] and close[i] > sma_50_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: momentum down + downtrend
                elif close[i] < sma_20_1h[i] and close[i] < sma_50_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals