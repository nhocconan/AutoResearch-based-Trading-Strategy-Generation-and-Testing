#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly pivot direction from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wweeklypivot_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    weekly_close = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:
            weekly_high[i] = np.max(high_1w[i-1:i])  # Current week's high
            weekly_low[i] = np.min(low_1w[i-1:i])    # Current week's low
            weekly_close[i] = close_1w[i-1]          # Previous week's close
        else:
            weekly_high[i] = high_1w[i]
            weekly_low[i] = low_1w[i]
            weekly_close[i] = close_1w[i]
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot and levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 14-period ATR on 1w for stoploss
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        atr_1w[0] = np.nan
        if len(tr) > 0:
            atr_1w[1] = tr[0]
            for i in range(2, len(atr_1w)):
                atr_1w[i] = (tr[i-1] * 13 + atr_1w[i-1]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 1)  # For Donchian and weekly data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on current price vs weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                close[i] < weekly_pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                close[i] > weekly_pivot_aligned[i] or
                close[i] > entry_price + 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and bullish_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and bearish_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly pivot direction from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wweeklypivot_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    weekly_close = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:
            weekly_high[i] = np.max(high_1w[i-1:i])  # Current week's high
            weekly_low[i] = np.min(low_1w[i-1:i])    # Current week's low
            weekly_close[i] = close_1w[i-1]          # Previous week's close
        else:
            weekly_high[i] = high_1w[i]
            weekly_low[i] = low_1w[i]
            weekly_close[i] = close_1w[i]
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot and levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 14-period ATR on 1w for stoploss
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        atr_1w[0] = np.nan
        if len(tr) > 0:
            atr_1w[1] = tr[0]
            for i in range(2, len(atr_1w)):
                atr_1w[i] = (tr[i-1] * 13 + atr_1w[i-1]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 1)  # For Donchian and weekly data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on current price vs weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                close[i] < weekly_pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                close[i] > weekly_pivot_aligned[i] or
                close[i] > entry_price + 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and bullish_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and bearish_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly pivot direction from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wweeklypivot_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    weekly_close = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:
            weekly_high[i] = np.max(high_1w[i-1:i])  # Current week's high
            weekly_low[i] = np.min(low_1w[i-1:i])    # Current week's low
            weekly_close[i] = close_1w[i-1]          # Previous week's close
        else:
            weekly_high[i] = high_1w[i]
            weekly_low[i] = low_1w[i]
            weekly_close[i] = close_1w[i]
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot and levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 14-period ATR on 1w for stoploss
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        atr_1w[0] = np.nan
        if len(tr) > 0:
            atr_1w[1] = tr[0]
            for i in range(2, len(atr_1w)):
                atr_1w[i] = (tr[i-1] * 13 + atr_1w[i-1]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 1)  # For Donchian and weekly data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on current price vs weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                close[i] < weekly_pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                close[i] > weekly_pivot_aligned[i] or
                close[i] > entry_price + 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and bullish_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and bearish_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly pivot direction from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wweeklypivot_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    weekly_close = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:
            weekly_high[i] = np.max(high_1w[i-1:i])  # Current week's high
            weekly_low[i] = np.min(low_1w[i-1:i])    # Current week's low
            weekly_close[i] = close_1w[i-1]          # Previous week's close
        else:
            weekly_high[i] = high_1w[i]
            weekly_low[i] = low_1w[i]
            weekly_close[i] = close_1w[i]
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot and levels to daily timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 14-period ATR on 1w for stoploss
    atr_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        tr = np.maximum(
            high_1w[1:] - low_1w[1:],
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
        atr_1w[0] = np.nan
        if len(tr) > 0:
            atr_1w[1] = tr[0]
            for i in range(2, len(atr_1w)):
                atr_1w[i] = (tr[i-1] * 13 + atr_1w[i-1]) / 14
    
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 1)  # For Donchian and weekly data
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine bias based on current price vs weekly pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        # Donchian channel (20-period)
        if i >= 20:
            highest_high = np.max(high[i-20:i])
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i+1]) if i > 0 else high[i]
            lowest_low = np.min(low[:i+1]) if i > 0 else low[i]
        
        # Volume filter (20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower OR below weekly pivot
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or 
                close[i] < weekly_pivot_aligned[i] or
                close[i] < entry_price - 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper OR above weekly pivot
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or 
                close[i] > weekly_pivot_aligned[i] or
                close[i] > entry_price + 2.0 * atr_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            bull_breakout = close[i] > highest_high
            bear_breakout = close[i] < lowest_low
            
            if i >= 20 and bull_breakout and volume_filter and bullish_bias:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif i >= 20 and bear_breakout and volume_filter and bearish_bias:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
1d Donchian(20) Breakout + 1w Weekly Pivot Direction + Volume Filter + ATR Stoploss
Hypothesis: Donchian breakouts capture momentum, weekly pivot direction from 1w timeframe filters for institutional bias, volume confirms breakout strength, ATR stoploss limits drawdown. Designed for low trade frequency (target 30-100 total over 4 years) to minimize fee decay. Works in bull/bear by only trading with higher timeframe pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1wweeklypivot_volume_atr_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for weekly pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    weekly_close = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        if i >= 1:
            weekly_high[i] = np.max(high_1w[i-1:i])  # Current week's high
            weekly_low[i] = np.min(low_1w[i-1:i])    # Current week's low
            weekly_close[i] = close_1w[i-1]          # Previous week's close
        else:
            weekly_high[i] = high_1w[i]
            weekly_low[i] = low_1w[i]
            weekly_close[i] = close_1w[i]
    
    # Weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + (weekly_high - weekly_low)
    s2 = weekly_pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot and levels to daily timeframe