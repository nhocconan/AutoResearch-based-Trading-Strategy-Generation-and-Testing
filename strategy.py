#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period 12h Donchian upper AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price breaks below 20-period 12h Donchian lower AND price < 1w EMA50 (strong downtrend) AND volume spike
# Donchian channels provide clear breakout levels with good risk/reward
# 1w EMA50 provides a robust long-term trend filter to reduce whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 12h (primary timeframe as required)

name = "12h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) on 12h timeframe
    if len(high) >= 20:
        # Rolling max/min for upper/lower bands
        upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        upper_20 = np.full(n, np.nan)
        lower_20 = np.full(n, np.nan)
    
    # Volume confirmation on 12h (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND strong uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > upper_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND strong downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < lower_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper Donchian OR closes below 1w EMA50
            if close[i] < upper_20[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower Donchian OR closes above 1w EMA50
            if close[i] > lower_20[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals