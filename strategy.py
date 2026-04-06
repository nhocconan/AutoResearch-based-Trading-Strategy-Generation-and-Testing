#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume and 1d Trend Alignment
Hypothesis: Donchian(20) breakouts aligned with 1d EMA200 trend, confirmed by volume spikes,
capture trend continuation moves while avoiding false breakouts in both bull and bear markets.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend alignment (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(200) for long-term trend
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require volume spike
    
    # 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: break below lower Donchian OR stoploss
            if (close[i] <= lowest_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above upper Donchian OR stoploss
            if (close[i] >= highest_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            long_breakout = close[i] > highest_high[i]
            short_breakout = close[i] < lowest_low[i]
            
            # Trend alignment: price vs 1d EMA200
            uptrend = close[i] > ema_200_1d_aligned[i]
            downtrend = close[i] < ema_200_1d_aligned[i]
            
            if long_breakout and uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume and 1d Trend Alignment
Hypothesis: Donchian(20) breakouts aligned with 1d EMA200 trend, confirmed by volume spikes,
capture trend continuation moves while avoiding false breakouts in both bull and bear markets.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data for trend alignment (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(200) for long-term trend
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)  # Require volume spike
    
    # 4h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 200  # For EMA200
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: break below lower Donchian OR stoploss
            if (close[i] <= lowest_low[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: break above upper Donchian OR stoploss
            if (close[i] >= highest_high[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume
            long_breakout = close[i] > highest_high[i]
            short_breakout = close[i] < lowest_low[i]
            
            # Trend alignment: price vs 1d EMA200
            uptrend = close[i] > ema_200_1d_aligned[i]
            downtrend = close[i] < ema_200_1d_aligned[i]
            
            if long_breakout and uptrend and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_breakout and downtrend and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals