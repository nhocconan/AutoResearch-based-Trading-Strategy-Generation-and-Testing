#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
# Donchian channel breakouts capture strong momentum moves. 1w EMA34 ensures alignment with weekly trend
# to avoid false breakouts in ranging markets. Volume spike confirms institutional participation.
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)
# by only taking trades in direction of 1w EMA34.

name = "1d_Donchian20_Breakout_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) channels using prior 20 periods (breakout lookback)
    # Use rolling window on prior data to avoid look-ahead
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Donchian and EMA)
    start_idx = max(40, 35)  # 20 for Donchian*2 + buffer, 34 for EMA + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_rolling[i]) or 
            np.isnan(low_rolling[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian with volume spike AND price > 1w EMA34 (bullish trend)
            if (close[i] > high_rolling[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian with volume spike AND price < 1w EMA34 (bearish trend)
            elif (close[i] < low_rolling[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below lower Donchian (failed breakout) OR price below 1w EMA34 (trend change)
            if close[i] < low_rolling[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian (failed breakdown) OR price above 1w EMA34 (trend change)
            if close[i] > high_rolling[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals