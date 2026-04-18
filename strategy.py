#!/usr/bin/env python3
"""
4h_VWAP_Reversion_1dTrend
4h strategy combining intraday VWAP mean reversion with daily trend filter.
- Long: Price crosses above VWAP + daily EMA50 > EMA200
- Short: Price crosses below VWAP + daily EMA50 < EMA200
- Exit: Opposite VWAP cross or trend reversal
Designed for ~20-40 trades/year per symbol (80-160 total over 4 years)
Works in bull markets (trend continuation) and bear markets (mean reversion within trend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for 4h data
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # VWAP cross conditions
        vwap_cross_up = close[i] > vwap[i] and close[i-1] <= vwap[i-1]
        vwap_cross_down = close[i] < vwap[i] and close[i-1] >= vwap[i-1]
        
        if position == 0:
            # Long: uptrend + VWAP cross up
            if uptrend and vwap_cross_up:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + VWAP cross down
            elif downtrend and vwap_cross_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change or VWAP cross down
            if not uptrend or vwap_cross_down:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change or VWAP cross up
            if not downtrend or vwap_cross_up:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Reversion_1dTrend"
timeframe = "4h"
leverage = 1.0