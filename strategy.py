#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation + ATR stoploss
# Long when price breaks above Donchian(20) high + price > EMA(50) + volume > 1.5x avg volume
# Short when price breaks below Donchian(20) low + price < EMA(50) + volume > 1.5x avg volume
# Exit when price crosses EMA(50) or ATR stoploss hit (2x ATR)
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA(50) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # ATR stoploss: exit if price moves 2*ATR against position
        if position == 1 and close[i] < entry_price - 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] > entry_price + 2.0 * atr[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Exit conditions: price crosses EMA(50)
        if position == 1 and close[i] < ema_50_aligned[i]:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] > ema_50_aligned[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: breakout above Donchian high + above EMA(50) + volume confirmation
            if close[i] > highest_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: breakout below Donchian low + below EMA(50) + volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
        else:
            # Hold position
            signals[i] = position * 0.25
    
    return signals