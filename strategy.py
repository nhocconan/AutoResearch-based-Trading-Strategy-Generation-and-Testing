#!/usr/bin/env python3
"""
6H DONCHIAN(20) BREAKOUT + 1D VOLUME CONFIRMATION + PULLBACK FILTER
Hypothesis: Donchian breakouts capture momentum bursts. Volume confirms breakout strength.
Pullback filter ensures we enter on retests of breakout level, reducing false breakouts.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_pullback_volume_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period) on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss hit
            if (close[i] < donchian_low[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss hit
            if (close[i] > donchian_high[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above Donchian high, then pulls back to retest the breakout level
            # We allow entry if price is within 0.5*ATR of the breakout level (Donchian high)
            if (close[i] > donchian_high[i-1] and  # broken above previous Donchian high
                abs(close[i] - donchian_high[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, then pulls back to retest the breakout level
            elif (close[i] < donchian_low[i-1] and  # broken below previous Donchian low
                  abs(close[i] - donchian_low[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER AND PULLBACK ENTRY
Hypothesis: Volatility expansions (ATR ratio > 1.5) signal breakout opportunities.
Enter on pullbacks to the breakout level with volume confirmation.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vol_breakout_pullback_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma = np.full(n, np.nan)
    for i in range(20, n):
        atr_ma[i] = np.mean(atr[i-20:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility expansion: ATR ratio > 1.5
        vol_expansion = atr_ratio[i] > 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 20-period low or stoploss hit
            if (close[i] < low_20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above 20-period high or stoploss hit
            if (close[i] > high_20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: volatility expansion + price breaks above 20-period high, then pulls back to retest
            if (vol_expansion and
                close[i] > high_20[i-1] and  # broken above previous 20-period high
                abs(close[i] - high_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: volatility expansion + price breaks below 20-period low, then pulls back to retest
            elif (vol_expansion and
                  close[i] < low_20[i-1] and  # broken below previous 20-period low
                  abs(close[i] - low_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER AND PULLBACK ENTRY
Hypothesis: Volatility expansions (ATR ratio > 1.5) signal breakout opportunities.
Enter on pullbacks to the breakout level with volume confirmation.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vol_breakout_pullback_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma = np.full(n, np.nan)
    for i in range(20, n):
        atr_ma[i] = np.mean(atr[i-20:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility expansion: ATR ratio > 1.5
        vol_expansion = atr_ratio[i] > 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 20-period low or stoploss hit
            if (close[i] < low_20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above 20-period high or stoploss hit
            if (close[i] > high_20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: volatility expansion + price breaks above 20-period high, then pulls back to retest
            if (vol_expansion and
                close[i] > high_20[i-1] and  # broken above previous 20-period high
                abs(close[i] - high_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: volatility expansion + price breaks below 20-period low, then pulls back to retest
            elif (vol_expansion and
                  close[i] < low_20[i-1] and  # broken below previous 20-period low
                  abs(close[i] - low_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER AND PULLBACK ENTRY
Hypothesis: Volatility expansions (ATR ratio > 1.5) signal breakout opportunities.
Enter on pullbacks to the breakout level with volume confirmation.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vol_breakout_pullback_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma = np.full(n, np.nan)
    for i in range(20, n):
        atr_ma[i] = np.mean(atr[i-20:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility expansion: ATR ratio > 1.5
        vol_expansion = atr_ratio[i] > 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 20-period low or stoploss hit
            if (close[i] < low_20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above 20-period high or stoploss hit
            if (close[i] > high_20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: volatility expansion + price breaks above 20-period high, then pulls back to retest
            if (vol_expansion and
                close[i] > high_20[i-1] and  # broken above previous 20-period high
                abs(close[i] - high_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: volatility expansion + price breaks below 20-period low, then pulls back to retest
            elif (vol_expansion and
                  close[i] < low_20[i-1] and  # broken below previous 20-period low
                  abs(close[i] - low_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER AND PULLBACK ENTRY
Hypothesis: Volatility expansions (ATR ratio > 1.5) signal breakout opportunities.
Enter on pullbacks to the breakout level with volume confirmation.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vol_breakout_pullback_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma = np.full(n, np.nan)
    for i in range(20, n):
        atr_ma[i] = np.mean(atr[i-20:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility expansion: ATR ratio > 1.5
        vol_expansion = atr_ratio[i] > 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 20-period low or stoploss hit
            if (close[i] < low_20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above 20-period high or stoploss hit
            if (close[i] > high_20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: volatility expansion + price breaks above 20-period high, then pulls back to retest
            if (vol_expansion and
                close[i] > high_20[i-1] and  # broken above previous 20-period high
                abs(close[i] - high_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: volatility expansion + price breaks below 20-period low, then pulls back to retest
            elif (vol_expansion and
                  close[i] < low_20[i-1] and  # broken below previous 20-period low
                  abs(close[i] - low_20[i-1]) < 0.5 * atr[i] and  # pulled back to retest
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER AND PULLBACK ENTRY
Hypothesis: Volatility expansions (ATR ratio > 1.5) signal breakout opportunities.
Enter on pullbacks to the breakout level with volume confirmation.
Works in bull (breakouts with trend) and bear (breakouts against trend filtered by price action).
Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vol_breakout_pullback_v1"
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
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # ATR ratio: current ATR / 20-period average ATR (volatility expansion filter)
    atr_ma = np.full(n, np.nan)
    for i in range(20, n):
        atr_ma[i] = np.mean(atr[i-20:i])
    atr_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume average on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 20-period high/low for breakout levels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 1.5x 20-day average volume (aligned)
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility expansion: ATR ratio > 1.5
        vol_expansion = atr_ratio[i] > 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 20-period low or stoploss hit
            if (close[i] < low_20