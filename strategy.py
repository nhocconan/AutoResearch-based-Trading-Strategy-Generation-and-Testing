#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter.
# Long when price breaks above upper Donchian channel + volume spike + price > 12h EMA50 (bull trend).
# Short when price breaks below lower Donchian channel + volume spike + price < 12h EMA50 (bear trend).
# Exit when price crosses the opposite Donchian channel or volume dries up.
# Designed for 75-200 total trades over 4 years (19-50/year) with strong risk-adjusted returns.

name = "4h_Donchian20_VolumeSpike_12hEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        close_val = close[i]
        up_chan = upper_channel[i]
        low_chan = lower_channel[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(up_chan) or np.isnan(low_chan) or np.isnan(ema_trend):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Breakout conditions
        long_breakout = close_val > up_chan
        short_breakout = close_val < low_chan
        
        # Generate signals
        if position == 0:
            if long_breakout and vol_spike and is_bull_trend:
                signals[i] = 0.30
                position = 1
            elif short_breakout and vol_spike and is_bear_trend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit on breakdown of lower channel or loss of bull trend
            if close_val < low_chan or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit on breakout of upper channel or loss of bear trend
            if close_val > up_chan or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals