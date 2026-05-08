#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR breakout with volume confirmation and 1w EMA trend filter.
# Uses 1d ATR to define breakout levels from recent highs/lows.
# Long when price breaks above recent high + 1.5*1d ATR with volume surge and above 1w EMA.
# Short when price breaks below recent low - 1.5*1d ATR with volume surge and below 1w EMA.
# Designed for low trade frequency (20-40/year) to avoid fee drift. ATR breakouts capture volatility expansion.
# Works in both bull and bear markets by filtering with 1w EMA trend and requiring volume confirmation.

name = "4h_1dATRBreakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # First period has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr_1d = np.zeros_like(tr)
    atr_1d[0] = tr[0]
    for i in range(1, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate 1w EMA (using 5 days as proxy for 1 week)
    ema_1w = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_1w)
    
    # Calculate recent high/low for breakout levels (using 10 periods lookback)
    # We'll use the highest high and lowest low from the last 10 periods
    lookback = 10
    recent_high = np.full(n, np.nan)
    recent_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        recent_high[i] = np.max(high[i-lookback:i])
        recent_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: 4h volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or 
            np.isnan(recent_high[i]) or 
            np.isnan(recent_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Calculate breakout levels
            upper_break = recent_high[i] + 1.5 * atr_1d_aligned[i]
            lower_break = recent_low[i] - 1.5 * atr_1d_aligned[i]
            
            # Enter long: price breaks above upper level + volume surge + above 1w EMA
            if close[i] > upper_break and vol_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower level + volume surge + below 1w EMA
            elif close[i] < lower_break and vol_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below recent low (trailing stop)
            if close[i] < recent_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above recent high (trailing stop)
            if close[i] > recent_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals