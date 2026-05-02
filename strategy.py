#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses daily EMA34 for trend direction (price > EMA34 = uptrend, price < EMA34 = downtrend)
# Entry logic: Break above Donchian(20) high in uptrend with volume spike for long
#              Break below Donchian(20) low in downtrend with volume spike for short
# Exit: Opposite Donchian(10) break or trend reversal (close crosses EMA34)
# Designed to work in both bull and bear markets by trading with the daily trend
# Target: 12-37 trades/year (50-150 total over 4 years) for 12h timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "12h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Donchian(20) channels
    # Need at least 20 bars for calculation
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h Donchian(10) for exit (tighter channel)
    high_ma_exit = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_ma_exit = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above Donchian(20) high AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below Donchian(20) low AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below Donchian(10) low OR price < 1d EMA34 (trend change)
            if (close[i] < low_ma_exit[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above Donchian(10) high OR price > 1d EMA34 (trend change)
            if (close[i] > high_ma_exit[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals