#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 4h, filtered by 12h trend direction (HMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v2"
timeframe = "4h"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on 12h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
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
4h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation v2
Hypothesis: Breakouts from Donchian channels on 4h, filtered by 12h trend direction (HMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Tightening volume filter to 3.0x to reduce trades and improve quality.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v3"
timeframe = "4h"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on 12h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 3.0x average over last 20 periods (increased from 2.0x)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 3.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
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
4h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation v3
Hypothesis: Breakouts from Donchian channels on 4h, filtered by 12h trend direction (HMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Tightening volume filter to 4.0x to reduce trades and improve quality.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v4"
timeframe = "4h"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on 12h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 4.0x average over last 20 periods (increased from 3.0x)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 4.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
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
4h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation v5
Hypothesis: Breakouts from Donchian channels on 4h, filtered by 12h trend direction (HMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Tightening volume filter to 5.0x to reduce trades and improve quality.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v5"
timeframe = "4h"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on 12h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 5.0x average over last 20 periods (increased from 4.0x)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 5.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
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
4h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation v6
Hypothesis: Breakouts from Donchian channels on 4h, filtered by 12h trend direction (HMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Tightening volume filter to 6.0x to reduce trades and improve quality.
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian20_12h_trend_vol_v6"
timeframe = "4h"
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
    
    # 20-period ATR for stops and filters
    atr = np.full(n, np.nan)
    if n >= 20:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[20] = np.mean(tr[:20])
            for i in range(21, n):
                atr[i] = (atr[i-1] * 19 + tr[i-1]) / 20
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate Hull Moving Average (HMA) for trend on 12h
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(half, len(arr)):
            wma2[i] = np.nansum(arr[i-half+1:i+1] * np.arange(1, half+1)) / (half * (half + 1) / 2)
        for i in range(period, len(arr)):
            wma1[i] = np.nansum(arr[i-period+1:i+1] * np.arange(1, period+1)) / (period * (period + 1) / 2)
        hma_raw = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(sqrt, len(arr)):
            hma_result[i] = np.nansum(hma_raw[i-sqrt+1:i+1] * np.arange(1, sqrt+1)) / (sqrt * (sqrt + 1) / 2)
        return hma_result
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Determine trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
    trend_12h = np.where(close_12h > hma_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 6.0x average over last 20 periods (increased from 5.0x)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 6.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or