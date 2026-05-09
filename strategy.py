#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d trend filter.
# Long when price breaks above 4h Donchian upper band, volume > 1.5x 20-period average, and price > 1d EMA50.
# Short when price breaks below 4h Donchian lower band, volume > 1.5x 20-period average, and price < 1d EMA50.
# Exit when price crosses back below/above the opposite Donchian band OR 1d EMA50 direction contradicts position.
# Session filter: only trade between 08:00-20:00 UTC to avoid low-volume periods.
# Position size: 0.20 (20% of capital) to limit drawdown and reduce trade frequency.
# Uses 4h for signal direction (Donchian breakout), 1h only for entry timing and session filter.

name = "1h_DonchianBreakout_Volume_1dEMA50_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels: 20-period high and low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (waits for 4h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_1d[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + volume spike + price > 1d EMA50
            if (close[i] > donchian_high_aligned[i] and 
                vol_spike[i] and 
                close[i] > ema50_1d[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low + volume spike + price < 1d EMA50
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_spike[i] and 
                  close[i] < ema50_1d[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 4h Donchian low OR price < 1d EMA50
            if (close[i] < donchian_low_aligned[i]) or (close[i] < ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above 4h Donchian high OR price > 1d EMA50
            if (close[i] > donchian_high_aligned[i]) or (close[i] > ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals