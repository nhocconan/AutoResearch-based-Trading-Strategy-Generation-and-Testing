#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume spike and 1d EMA34 trend filter
# Uses Donchian breakout as primary signal, filtered by 1d EMA34 trend and volume confirmation.
# Works in both bull and bear by following daily trend direction.
# Target: 15-25 trades/year per symbol (60-100 total) to avoid excessive fees.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donchian_high = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 12-hour timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume spike + uptrend (price > 1d EMA34)
            if (close[i] > donchian_high_aligned[i] and vol_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume spike + downtrend (price < 1d EMA34)
            elif (close[i] < donchian_low_aligned[i] and vol_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_Spike_1dEMA34_Trend"
timeframe = "12h"
leverage = 1.0