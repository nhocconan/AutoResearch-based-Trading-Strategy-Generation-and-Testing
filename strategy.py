#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray12h_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(13) for trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = close_12h.copy()
    alpha = 2.0 / (13 + 1)
    ema_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12-hour EMA(13) for high and low (Elder Ray components)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    ema_high_12h = high_12h.copy()
    ema_low_12h = low_12h.copy()
    for i in range(1, len(close_12h)):
        ema_high_12h[i] = alpha * high_12h[i] + (1 - alpha) * ema_high_12h[i-1]
        ema_low_12h[i] = alpha * low_12h[i] + (1 - alpha) * ema_low_12h[i-1]
    ema_high_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_high_12h)
    ema_low_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_low_12h)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_12h - ema_high_12h
    bear_power = low_12h - ema_low_12h
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # 12-hour volume average for confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(19, len(vol_12h)):  # 20-period average
        vol_ma_12h[i] = np.mean(vol_12h[i-19:i+1])
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Need 20 for volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 12h average
        volume_filter = volume[i] > vol_ma_12h_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns negative or stoploss
            if (bear_power_aligned[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns positive or stoploss
            if (bull_power_aligned[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: Bull Power positive and Bear Power negative (bullish)
                if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bull Power negative and Bear Power positive (bearish)
                elif bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray12h_trend_vol_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(13) for trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = close_12h.copy()
    alpha = 2.0 / (13 + 1)
    ema_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_12h[i-1]
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12-hour EMA(13) for high and low (Elder Ray components)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    ema_high_12h = high_12h.copy()
    ema_low_12h = low_12h.copy()
    for i in range(1, len(close_12h)):
        ema_high_12h[i] = alpha * high_12h[i] + (1 - alpha) * ema_high_12h[i-1]
        ema_low_12h[i] = alpha * low_12h[i] + (1 - alpha) * ema_low_12h[i-1]
    ema_high_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_high_12h)
    ema_low_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_low_12h)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_12h - ema_high_12h
    bear_power = low_12h - ema_low_12h
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    # 12-hour volume average for confirmation
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    for i in range(19, len(vol_12h)):  # 20-period average
        vol_ma_12h[i] = np.mean(vol_12h[i-19:i+1])
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20  # Need 20 for volume MA
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 12h average
        volume_filter = volume[i] > vol_ma_12h_aligned[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns negative or stoploss
            if (bear_power_aligned[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns positive or stoploss
            if (bull_power_aligned[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: Bull Power positive and Bear Power negative (bullish)
                if bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bull Power negative and Bear Power positive (bearish)
                elif bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals