#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: Breakouts from weekly Donchian channels filtered by 1w trend (EMA crossover)
and confirmed by volume spikes capture major trend moves while avoiding whipsaws.
Works in both bull and bear markets by following the dominant weekly trend.
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period high/low) on daily
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA for trend on 1w
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_1w, 9)
    ema_slow = ema(close_1w, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_1w = np.where(ema_fast > ema_slow, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
           np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma[i]):
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
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 1w trend with volume
            if (close[i] > donch_high[i] and
                trend_1w_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 1w trend with volume
            elif (close[i] < donch_low[i] and
                  trend_1w_aligned[i] == -1 and
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
1d Donchian(20) Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: Breakouts from weekly Donchian channels filtered by 1w trend (EMA crossover)
and confirmed by volume spikes capture major trend moves while avoiding whipsaws.
Works in both bull and bear markets by following the dominant weekly trend.
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period high/low) on daily
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA for trend on 1w
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_1w, 9)
    ema_slow = ema(close_1w, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_1w = np.where(ema_fast > ema_slow, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
           np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma[i]):
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
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 1w trend with volume
            if (close[i] > donch_high[i] and
                trend_1w_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 1w trend with volume
            elif (close[i] < donch_low[i] and
                  trend_1w_aligned[i] == -1 and
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
1d Donchian(20) Breakout with 1w Trend Filter and Volume Confirmation
Hypothesis: Breakouts from weekly Donchian channels filtered by 1w trend (EMA crossover)
and confirmed by volume spikes capture major trend moves while avoiding whipsaws.
Works in both bull and bear markets by following the dominant weekly trend.
Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period high/low) on daily
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA for trend on 1w
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_fast = ema(close_1w, 9)
    ema_slow = ema(close_1w, 21)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1w, ema_slow)
    
    # Determine trend: 1 if fast EMA > slow EMA (bullish), -1 if fast EMA < slow EMA (bearish)
    trend_1w = np.where(ema_fast > ema_slow, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
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
           np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ma[i]):
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
                trend_1w_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_1w_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in bullish 1w trend with volume
            if (close[i] > donch_high[i] and
                trend_1w_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in bearish 1w trend with volume
            elif (close[i] < donch_low[i] and
                  trend_1w_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

--- End of file ---