#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot strategy with 1d trend filter and volume confirmation
# Long when price breaks above daily R4 with daily EMA(20) rising and volume > 20-period average
# Short when price breaks below daily S4 with daily EMA(20) falling and volume > 20-period average
# Uses 6h timeframe with 1d pivot levels and trend filter to reduce whipsaw in choppy markets
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets

name = "6h_camarilla1d_ema_vol_v1"
timeframe = "6h"
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
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(20) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate daily Camarilla pivot levels (based on previous day)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # S4 = C - ((H-L) * 1.1/2)
    # We use previous day's values to avoid look-ahead
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    s4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below daily S4 or 1d EMA turns down
            elif close[i] < s4_1d_aligned[i] or ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above daily R4 or 1d EMA turns up
            elif close[i] > r4_1d_aligned[i] or ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if vol_filter[i]:
                # Long when price breaks above daily R4 and 1d EMA rising
                if close[i] > r4_1d_aligned[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when price breaks below daily S4 and 1d EMA falling
                elif close[i] < s4_1d_aligned[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals