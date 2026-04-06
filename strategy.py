#!/usr/bin/env python3
"""
12H DYNAMIC SUPPORT/RESISTANCE WITH VOLUME FILTER
Hypothesis: Price tends to respect dynamic support/resistance levels derived from 
exponential moving averages on higher timeframes. In bull markets, price finds 
support at rising EMAs; in bear markets, resistance at falling EMAs. Volume 
confirmation filters out false breaks. Weekly EMA trend filter ensures alignment 
with major trend. Designed for fewer trades (<50/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_dynamic_sr_volume_filter"
timeframe = "12h"
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
    
    # ATR for stops
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(20) for trend
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_weekly = ema(close_weekly, 20)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Get daily data for dynamic S/R
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Calculate daily EMA(50) as dynamic support/resistance
    ema_daily = ema(close_daily, 50)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below dynamic support or stoploss hit
            if (close[i] < ema_daily_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above dynamic resistance or stoploss hit
            if (close[i] > ema_daily_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price above dynamic support, above weekly EMA trend, with volume
            if (close[i] > ema_daily_aligned[i] and 
                close[i] > ema_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below dynamic resistance, below weekly EMA trend, with volume
            elif (close[i] < ema_daily_aligned[i] and 
                  close[i] < ema_weekly_aligned[i] and 
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
12H WEEKLY PIVOT REVERSAL WITH VOLUME CONFIRMATION
Hypothesis: Weekly pivot points act as significant support/resistance levels. 
Price often reverses at these levels, especially when aligned with the weekly 
trend (EMA40). Volume confirmation filters out false signals. Works in both 
bull and bear markets by trading reversals at key levels. Target: 50-150 
trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_weekly_pivot_reversal"
timeframe = "12h"
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
    
    # ATR for stops
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
    
    # Get weekly data for pivot points and trend
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Support 1: S1 = 2*P - H
    # Resistance 1: R1 = 2*P - L
    pivot_point = (high_weekly + low_weekly + close_weekly) / 3.0
    support_1 = 2 * pivot_point - high_weekly
    resistance_1 = 2 * pivot_point - low_weekly
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot_point)
    support_1_aligned = align_htf_to_ltf(prices, df_weekly, support_1)
    resistance_1_aligned = align_htf_to_ltf(prices, df_weekly, resistance_1)
    
    # Weekly EMA(40) for trend filter
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_weekly = ema(close_weekly, 40)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter: current volume > 1.4x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(pivot_aligned[i]) or np.isnan(support_1_aligned[i]) or np.isnan(resistance_1_aligned[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.4
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below support or stoploss hit
            if (close[i] < support_1_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above resistance or stoploss hit
            if (close[i] > resistance_1_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries at pivot reversals
            # Long: price bounces off support, above weekly EMA trend, with volume
            if (close[i] > support_1_aligned[i] and 
                low[i] <= support_1_aligned[i] * 1.005 and  # touched support
                close[i] > ema_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price rejects resistance, below weekly EMA trend, with volume
            elif (close[i] < resistance_1_aligned[i] and 
                  high[i] >= resistance_1_aligned[i] * 0.995 and  # touched resistance
                  close[i] < ema_weekly_aligned[i] and 
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
12H KAMA + RSI + VOLUME FILTER
Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise, 
providing a smooth trend line that whipsaws less in choppy markets. Combined 
with RSI for overbought/oversold conditions and volume confirmation, this 
strategy captures trend continuations while avoiding false signals. Weekly 
EMA filter ensures alignment with major trend. Designed for low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stops
    high = prices['high'].values
    low = prices['low'].values
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(20) for trend
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_weekly = ema(close_weekly, 20)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate KAMA(10, 2, 30)
    def kama(close, slow=10, fast=2):
        if len(close) < slow:
            return np.full_like(close, np.nan)
        
        # Efficiency Ratio
        change = np.abs(close - np.roll(close, slow))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        er = np.zeros_like(close)
        for i in range(slow, len(close)):
            if np.sum(np.abs(np.diff(close[i-slow+1:i+1]))) > 0:
                er[i] = change[i] / np.sum(np.abs(np.diff(close[i-slow+1:i+1])))
            else:
                er[i] = 0
        
        # Smoothing constant
        sc = np.square(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))
        
        # KAMA
        kama_val = np.full_like(close, np.nan)
        kama_val[slow] = np.mean(close[:slow])
        for i in range(slow+1, len(close)):
            kama_val[i] = kama_val[i-1] + sc[i] * (close[i] - kama_val[i-1])
        return kama_val
    
    kama_val = kama(close, 10, 2)
    
    # Calculate RSI(14)
    def rsi(close, period=14):
        if len(close) < period + 1:
            return np.full_like(close, np.nan)
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan)
        avg_loss = np.full_like(close, np.nan)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_val = rsi(close, 14)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(kama_val[i]) or np.isnan(rsi_val[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below KAMA or stoploss hit
            if (close[i] < kama_val[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above KAMA or stoploss hit
            if (close[i] > kama_val[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price above KAMA, RSI > 50 (bullish momentum), with volume
            if (close[i] > kama_val[i] and 
                rsi_val[i] > 50 and 
                close[i] > ema_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price below KAMA, RSI < 50 (bearish momentum), with volume
            elif (close[i] < kama_val[i] and 
                  rsi_val[i] < 50 and 
                  close[i] < ema_weekly_aligned[i] and 
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
12H VOLATILITY-BASED BREAKOUT WITH VOLUME FILTER
Hypothesis: Volatility contractions (low ATR) often precede expansion phases. 
When price breaks out of a volatility contraction with volume confirmation, 
it signals the start of a new trend. Weekly EMA filter ensures we only trade 
in the direction of the major trend. Designed for low frequency to minimize 
fee drag in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout"
timeframe = "12h"
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
    
    # ATR for stops and volatility measurement
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(20) for trend
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_weekly = ema(close_weekly, 20)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate ATR ratio: current ATR / average ATR over last 50 periods
    # Low ratio indicates volatility contraction
    atr_ma = np.full(n, np.nan)
    for i in range(50, n):
        atr_ma[i] = np.mean(atr[i-50:i])
    
    atr_ratio = np.full(n, np.nan)
    for i in range(50, n):
        if atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Calculate Donchian channels (20-period) for breakout levels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Volatility contraction filter: ATR ratio < 0.8 (low volatility)
        vol_contract = atr_ratio[i] < 0.8
        
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
            # Look for breakout entries
            # Long: price breaks above Donchian high, volatility contracting, above weekly EMA, with volume
            if (close[i] > donchian_high[i] and 
                vol_contract and 
                close[i] > ema_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low, volatility contracting, below weekly EMA, with volume
            elif (close[i] < donchian_low[i] and 
                  vol_contract and 
                  close[i] < ema_weekly_aligned[i] and 
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
12H TRIX + VOLUME + TREND FILTER
Hypothesis: TRIX (Triple Exponential Average) filters out insignificant price 
movements and shows the smooth underlying trend. When TRIX crosses zero with 
volume confirmation and aligns with weekly EMA trend, it signals a reliable 
trend change. Designed for low frequency to avoid whipsaws in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR for stops
    high = prices['high'].values
    low = prices['low'].values
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(20) for trend
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_weekly = ema(close_weekly, 20)
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate TRIX(12, 9)
    # TRIX = EMA(EMA(EMA(close, 12), 9), 9)
    def ema_series(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    # First EMA
    ema1 = ema_series(close, 12)
    # Second EMA
    ema2 = ema_series(ema1, 9)
    # Third EMA
    ema3 = ema_series(ema2, 9)
    
    # Calculate TRIX: percentage change of ema3
    trix = np.full_like(close, np.nan)
    for i in range(1, len(close)):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Volume filter: current volume > 1.3x average over last 20 periods
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
        if np.isnan(atr[i]) or np.isnan(trix[i]) or np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: TRIX crosses below zero or stoploss hit
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               (close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: TRIX crosses above zero or stoploss hit
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               (close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: TRIX crosses above zero, above weekly EMA trend, with volume
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                close[i] > ema_weekly_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: TRIX crosses below zero, below weekly EMA trend, with volume
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  close[i] < ema_weekly_aligned[i] and 
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
12H VOLUME WEIGHTED AVERAGE PRICE (VWAP) DEVIATION WITH VOLUME FILTER
Hypothesis: Price tends to revert to the VWAP, especially when extended 
beyond 1.5 standard deviations. Weekly EMA filter ensures we only take mean 
reversion trades in the direction of the major trend. Volume confirmation 
filters out low-probability setups. Designed for low frequency to minimize 
fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_vwap_deviation"
timeframe = "12h"
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
    
    # Typical price
    typical_price = (high + low + close) / 3.0
    
    # ATR for stops
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate