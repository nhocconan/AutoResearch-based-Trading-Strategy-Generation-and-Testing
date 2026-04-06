#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
# Long: price > Donchian upper (20) AND price > 12h EMA(50) AND volume > 1.5x volume MA(20)
# Short: price < Donchian lower (20) AND price < 12h EMA(50) AND volume > 1.5x volume MA(20)
# Exit: opposite Donchian break or price crosses 12h EMA(50)
# ATR(10) stop loss: exit if price moves 2.5*ATR against position
# Designed to capture trends with institutional volume confirmation, works in bull/bear.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_donchian20_12h_ema50_vol_v1"
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
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR(10) for stop loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first period
    atr = pd.Series(tr).ewm(span=10, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price < Donchian lower OR price < EMA50 OR stop loss hit
            if (close[i] < low_min[i] or close[i] < ema_50_aligned[i] or 
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price > Donchian upper OR price > EMA50 OR stop loss hit
            if (close[i] > high_max[i] or close[i] > ema_50_aligned[i] or 
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + EMA50 trend + volume filter
            if vol_filter[i]:
                if close[i] > high_max[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout with volume and trend: long
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif close[i] < low_min[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakdown with volume and trend: short
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals