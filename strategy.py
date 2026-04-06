#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation
Hypothesis: On the daily timeframe, breakouts of the 20-day Donchian channel aligned
with the weekly trend (EMA50) and confirmed by volume spikes capture significant moves
while avoiding false breakouts. Works in bull markets (long breakouts) and bear markets
(short breakdowns). Uses volatility-based exits with ATR to manage risk.
Target: 75-250 total trades over 4 years (19-62/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average True Range for volatility and stops
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 50  # For Donchian and EMA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price closes below midpoint of channel OR stoploss
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] <= midpoint or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above midpoint of channel OR stoploss
            midpoint = (highest_high[i] + lowest_low[i]) / 2
            if (close[i] >= midpoint or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend + volume confirmation
            long_breakout = close[i] > highest_high[i-1]  # Break above prior high
            short_breakout = close[i] < lowest_low[i-1]   # Break below prior low
            
            uptrend = ema_50_1w_aligned[i] > close[i]  # Price above weekly EMA = uptrend
            downtrend = ema_50_1w_aligned[i] < close[i]  # Price below weekly EMA = downtrend
            
            vol_filter = volume[i] > (2.0 * vol_ma[i])  # Require strong volume spike
            
            if long_breakout and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals