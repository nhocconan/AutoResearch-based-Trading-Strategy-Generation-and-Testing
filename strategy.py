#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, filtered by 12h trend direction (EMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA for trend on 12h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_12h, 9)
    ema_slow = ema(close_12h, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_12h = np.where(ema_fast > ema_slow, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
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
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, filtered by 12h trend direction (EMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA for trend on 12h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_12h, 9)
    ema_slow = ema(close_12h, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_12h = np.where(ema_fast > ema_slow, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
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
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, filtered by 12h trend direction (EMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA for trend on 12h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_12h, 9)
    ema_slow = ema(close_12h, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_12h = np.where(ema_fast > ema_slow, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
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
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, filtered by 12h trend direction (EMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA for trend on 12h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_12h, 9)
    ema_slow = ema(close_12h, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_12h = np.where(ema_fast > ema_slow, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
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
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume
            elif (close[i] < donch_low[i] and
                  trend_12h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
6h Donchian(20) Breakout with 12h Trend Filter and Volume Confirmation
Hypothesis: Breakouts from Donchian channels on 6h, filtered by 12h trend direction (EMA crossover),
and confirmed by volume spikes, capture momentum across market regimes. Using 12h trend
avoids whipsaws in sideways markets while capturing trends in both bull and bear phases.
Volume ensures breakout legitimacy. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_12h_trend_vol_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA for trend on 12h
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_12h, 9)
    ema_slow = ema(close_12h, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_12h = np.where(ema_fast > ema_slow, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 2.0x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
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
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_12h_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_12h_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 12h trend with volume
            if (close[i] > donch_high[i] and
                trend_12h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 12h trend with volume