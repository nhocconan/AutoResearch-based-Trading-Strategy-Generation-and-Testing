#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 12h EMA(50), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 12h EMA(50), volume > 1.5x avg
# Exit when: opposite Donchian break occurs or trailing stop hit (2*ATR)
# Uses 12h trend to filter false breakouts, targeting 75-200 trades over 4 years

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR for trailing stop (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trailing stop hit
            if close[i] < low_roll[i] or close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                # Trail stop: move up as price increases
                stop_price = max(stop_price, close[i] - 2.0 * atr[i])
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trailing stop hit
            if close[i] > high_roll[i] or close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                # Trail stop: move down as price decreases
                stop_price = min(stop_price, close[i] + 2.0 * atr[i])
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_roll[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above resistance with uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    stop_price = close[i] - 2.0 * atr[i]
                elif close[i] < low_roll[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below support with downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    stop_price = close[i] + 2.0 * atr[i]
    
    return signals