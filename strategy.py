#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily EMA filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 1.5x average
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 1.5x average
# Exit when price returns to Donchian midpoint or opposite breakout occurs
# Target: 75-200 trades over 4 years with controlled risk via 2*ATR stop

name = "4h_donchian20_1dema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR for dynamic stop (not used in signal but for exit logic)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                # Maintain position if already in trade
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR opposite breakout OR stoploss
            if (close[i] <= donchian_mid[i] or 
                low[i] <= high_max[i-1] or  # breakdown below previous Donchian high (failed breakout)
                (entry_price > 0 and close[i] < entry_price - 2.0 * atr[i])):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR opposite breakout OR stoploss
            if (close[i] >= donchian_mid[i] or 
                high[i] >= low_min[i-1] or  # breakout above previous Donchian low (failed breakdown)
                (entry_price > 0 and close[i] > entry_price + 2.0 * atr[i])):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_max[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above Donchian high with daily uptrend
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_min[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakdown below Donchian low with daily downtrend
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals