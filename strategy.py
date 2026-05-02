#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe to capture medium-term moves with low trade frequency
# Donchian breakouts capture momentum; 1d EMA34 ensures trend alignment
# Volume filter avoids low-validity breakouts
# Works in both bull and bear markets via trend-following logic
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
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
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels: upper = rolling max(high, 20), lower = rolling min(low, 20)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA34 (bullish trend) AND volume > 20-period average
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > vol_ma_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < 1d EMA34 (bearish trend) AND volume > 20-period average
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > vol_ma_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below 1d EMA34 (trend change)
            if close[i] < donch_low_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above 1d EMA34 (trend change)
            if close[i] > donch_high_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals