#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels capture volatility-based breakouts. Weekly EMA50 provides robust trend filter
# that works in both bull and bear markets. Volume confirmation ensures breakout conviction.
# Designed for 12-30 trades/year on 6h to minimize fee drag while maintaining edge.
# Uses discrete position sizing (0.25) to reduce churn and manage drawdown.

name = "6h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for EMA50 and Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using data up to current bar
        lookback = min(20, i+1)
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # Volume spike detection: current volume > 2.0 * 20-period volume EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (2.0 * vol_ema_20)
        
        # Breakout conditions
        breakout_long = close[i] > highest_high and volume_spike
        breakout_short = close[i] < lowest_low and volume_spike
        
        if position == 0:
            # Long: break above upper channel in 1w uptrend with volume spike
            if breakout_long and ema_50_1w_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel in 1w downtrend with volume spike
            elif breakout_short and ema_50_1w_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper channel or loses 1w uptrend
            if close[i] < highest_high or ema_50_1w_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower channel or loses 1w downtrend
            if close[i] > lowest_low or ema_50_1w_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals