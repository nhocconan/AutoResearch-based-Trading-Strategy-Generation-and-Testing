#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter, volume confirmation, and ATR stop loss
# Enters long on Donchian(20) breakout above when price > 12h EMA(50), volume > 1.5x average, and ATR volatility filter
# Enters short on Donchian(20) breakdown below when price < 12h EMA(50), volume > 1.5x average, and ATR volatility filter
# Exits on opposite Donchian touch or ATR-based stop loss
# Targets 75-200 trades over 4 years (19-50/year) with focus on trend continuation in both bull and bear markets

name = "4h_donchian20_12h_ema_vol_atrstop_v1"
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
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # ATR(14) for volatility filter and stop loss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(abs(high - pd.Series(close).shift(1)))
    tr3 = pd.Series(abs(low - pd.Series(close).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                # Maintain position until exit conditions
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stop loss
        if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 1:  # long position
            # Exit: price touches or crosses below Donchian lower band
            if close[i] <= low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price touches or crosses above Donchian upper band
            if close[i] >= high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume + volatility filter
            # Only enter when ATR is not too low (avoid choppy markets)
            if atr[i] > 0.01 * close[i]:  # Avoid extremely low volatility
                # Long entry: price breaks above Donchian upper band
                if (close[i] > high_max[i] and 
                    close[i] > ema_50_aligned[i] and 
                    volume[i] > volume_threshold[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian lower band
                elif (close[i] < low_min[i] and 
                      close[i] < ema_50_aligned[i] and 
                      volume[i] > volume_threshold[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals