#!/usr/bin/env python3
"""
6h Donchian Breakout + 12h Volatility Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum bursts; 12h volatility filter (ATR ratio) confirms genuine breakouts vs false moves, reducing whipsaw in chop. Works in bull/bear by capturing directional momentum with volatility confirmation. Target: 80-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hvol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # === Multi-timeframe: 12h ATR for volatility regime filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 30:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        # 14-period ATR on 12h
        atr_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 14:
            tr_12h = np.maximum(
                high_12h[1:] - low_12h[1:],
                np.abs(high_12h[1:] - close_12h[:-1]),
                np.abs(low_12h[1:] - close_12h[:-1])
            )
            atr_12h[0] = np.nan
            if len(tr_12h) > 0:
                atr_12h[1] = tr_12h[0]
                for i in range(2, len(df_12h)):
                    atr_12h[i] = (tr_12h[i-1] * 13 + atr_12h[i-1]) / 14
        # 50-period average of ATR for normalization
        atr_ma_12h = np.full(len(df_12h), np.nan)
        for i in range(len(df_12h)):
            if i >= 49 and not np.isnan(atr_12h[i]):
                atr_ma_12h[i] = np.mean(atr_12h[i-49:i+1])
        # Current ATR ratio: ATR(14) / MA(ATR,50)
        atr_ratio_12h = np.full(len(df_12h), np.nan)
        for i in range(len(df_12h)):
            if not np.isnan(atr_12h[i]) and not np.isnan(atr_ma_12h[i]) and atr_ma_12h[i] > 0:
                atr_ratio_12h[i] = atr_12h[i] / atr_ma_12h[i]
        # Align to 6s timeframe
        atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    else:
        atr_ratio_12h_aligned = np.full(n, 1.0)  # neutral if no data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # Donchian(20) and ATR MA(50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 0.8)
        vol_filter = atr_ratio_12h_aligned[i] > 0.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volatility filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if bull_breakout and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian Breakout + 12h Volatility Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum bursts; 12h volatility filter (ATR ratio) confirms genuine breakouts vs false moves, reducing whipsaw in chop. Works in bull/bear by capturing directional momentum with volatility confirmation. Target: 80-180 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12hvol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # === Multi-timeframe: 12h ATR for volatility regime filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 30:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        # 14-period ATR on 12h
        atr_12h = np.full(len(df_12h), np.nan)
        if len(df_12h) >= 14:
            tr_12h = np.maximum(
                high_12h[1:] - low_12h[1:],
                np.abs(high_12h[1:] - close_12h[:-1]),
                np.abs(low_12h[1:] - close_12h[:-1])
            )
            atr_12h[0] = np.nan
            if len(tr_12h) > 0:
                atr_12h[1] = tr_12h[0]
                for i in range(2, len(df_12h)):
                    atr_12h[i] = (tr_12h[i-1] * 13 + atr_12h[i-1]) / 14
        # 50-period average of ATR for normalization
        atr_ma_12h = np.full(len(df_12h), np.nan)
        for i in range(len(df_12h)):
            if i >= 49 and not np.isnan(atr_12h[i]):
                atr_ma_12h[i] = np.mean(atr_12h[i-49:i+1])
        # Current ATR ratio: ATR(14) / MA(ATR,50)
        atr_ratio_12h = np.full(len(df_12h), np.nan)
        for i in range(len(df_12h)):
            if not np.isnan(atr_12h[i]) and not np.isnan(atr_ma_12h[i]) and atr_ma_12h[i] > 0:
                atr_ratio_12h[i] = atr_12h[i] / atr_ma_12h[i]
        # Align to 6s timeframe
        atr_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ratio_12h)
    else:
        atr_ratio_12h_aligned = np.full(n, 1.0)  # neutral if no data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50)  # Donchian(20) and ATR MA(50)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 0.8)
        vol_filter = atr_ratio_12h_aligned[i] > 0.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volatility filter
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if bull_breakout and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif bear_breakout and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals