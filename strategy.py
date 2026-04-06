#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 20-period Donchian breakout with weekly trend filter and volume confirmation
# Enter long when: Price breaks above Donchian upper (20-day high), weekly EMA(50) up, volume > 1.5x average
# Enter short when: Price breaks below Donchian lower (20-day low), weekly EMA(50) down, volume > 1.5x average
# Exit on opposite Donchian break or trailing stop (2*ATR)
# Targets 50-100 trades over 4 years by requiring confluence of breakout, trend, and volume

name = "1d_donchian20_weekly_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-day)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_slope = ema_50 - np.roll(ema_50, 1)
    ema_50_slope[0] = 0
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    
    # Volume confirmation: volume > 1.5x 20-day average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR for stop loss (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower OR 2*ATR trailing stop
            if close[i] < donchian_lower[i] or close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper OR 2*ATR trailing stop
            if close[i] > donchian_upper[i] or close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + weekly trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_upper[i] and ema_slope_aligned[i] > 0:
                    # Breakout above resistance with up-trend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < donchian_lower[i] and ema_slope_aligned[i] < 0:
                    # Breakdown below support with down-trend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals