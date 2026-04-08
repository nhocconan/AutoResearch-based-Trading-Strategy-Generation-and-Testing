#!/usr/bin/env python3
"""
4h Donchian Breakout + 12h EMA Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves. Filtered by 12h EMA trend to ensure alignment with higher-timeframe momentum, volume spike to confirm breakout strength, and ATR-based stoploss to manage risk. This combination works in both bull and bear regimes by adapting to volatility and requiring multiple confirmations, reducing false breakouts and whipsaws. Targets 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(50) for trend filter
    ema_50_12h = df_12h['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(20) for stoploss and volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Volume filter (>2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend reverses or ATR-based stoploss
            if (close[i] <= lowest_low[i] or 
                close[i] < ema_50_12h_aligned[i] or
                close[i] <= (highest_high[i - 1] - 2.5 * atr[i])):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend reverses or ATR-based stoploss
            if (close[i] >= highest_high[i] or 
                close[i] > ema_50_12h_aligned[i] or
                close[i] >= (lowest_low[i - 1] + 2.5 * atr[i])):  # ATR trailing stop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Long breakout with trend alignment and volume
            if (close[i] >= highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.30
            # Short breakdown with trend alignment and volume
            elif (close[i] <= lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.30
    
    return signals