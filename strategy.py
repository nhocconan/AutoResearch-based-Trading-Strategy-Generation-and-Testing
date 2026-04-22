#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First day: use high-low only
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 40-period EMA on daily close for trend filter
    ema_40 = pd.Series(close_1d).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Donchian channel (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (15-period on 4h)
    vol_ma15 = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    vol_spike = volume > 2.0 * vol_ma15  # Require 2x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align daily indicators to 4-hour timeframe
    ema_40_aligned = align_htf_to_ltf(prices, df_1d, ema_40)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(ema_40_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma15[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above daily EMA40 + volume spike
            if (close[i] > highest_high[i] and close[i] > ema_40_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below daily EMA40 + volume spike
            elif (close[i] < lowest_low[i] and close[i] < ema_40_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back below/above daily EMA40
            if position == 1:
                if close[i] < ema_40_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_40_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA40_Trend_Volume_Spike"
timeframe = "4h"
leverage = 1.0