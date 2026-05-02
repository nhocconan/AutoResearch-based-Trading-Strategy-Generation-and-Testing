#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 for trend filter (long-term trend) and 1d Donchian channels for breakout signals
# Entry logic: Long when price breaks above 1d Donchian upper channel with volume spike and price > 1w EMA50
#              Short when price breaks below 1d Donchian lower channel with volume spike and price < 1w EMA50
# Exit logic: Exit when price crosses the 1w EMA50 (trend reversal) or opposite Donchian channel
# Works in both bull and bear markets by trading with the 1w trend
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Discrete sizing 0.25 balances profit potential and fee drag

name = "1d_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Break above 1d Donchian upper channel AND price > 1w EMA50 (uptrend) AND volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Break below 1d Donchian lower channel AND price < 1w EMA50 (downtrend) AND volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close below 1w EMA50 (trend change) OR break below 1d Donchian lower channel (reversal)
            if (close[i] < ema_50_1w_aligned[i] or 
                close[i] < low_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close above 1w EMA50 (trend change) OR break above 1d Donchian upper channel (reversal)
            if (close[i] > ema_50_1w_aligned[i] or 
                close[i] > high_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals