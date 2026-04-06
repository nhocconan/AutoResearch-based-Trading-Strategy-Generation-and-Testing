#!/usr/bin/env python3
"""
6h ADX + Volume Confirmation with 12h Trend Filter
Hypothesis: In trending markets (ADX > 25), price pulls back to the 20-period EMA offer high-probability entries.
The 12h EMA (50/200) filters the trend direction to avoid counter-trend trades. Volume confirms momentum.
Works in bull/bear by only trading with the higher timeframe trend. Target: 80-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_ema_volume_12h_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators ===
    # ADX (14) - trend strength
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([ [np.nan], tr ])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([ [0], dm_plus ])
        dm_minus = np.concatenate([ [0], dm_minus ])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # First average
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(dm_plus_smooth, np.nan)
        minus_di = np.full_like(dm_minus_smooth, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = (atr != 0) & ~np.isnan(atr)
        plus_di[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        minus_di[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx_sum = plus_di + minus_di
        valid_dx = (dx_sum != 0) & ~np.isnan(dx_sum)
        dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / dx_sum[valid_dx]
        
        # ADX - smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            valid_dx_first = ~np.isnan(dx[period:2*period+1])
            if np.any(valid_dx_first):
                first_valid_idx = np.where(valid_dx_first)[0][0] + period
                adx[first_valid_idx] = np.nanmean(dx[first_valid_idx-period+1:first_valid_idx+1])
                
                for i in range(first_valid_idx+1, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # EMA(20) for pullback entries
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema20 = ema(close, 20)
    
    # Volume filter: current volume > 1.5x average of last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) and EMA(200) for trend direction
    ema50_12h = ema(close_12h, 50)
    ema200_12h = ema(close_12h, 200)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Trend: 1 if EMA50 > EMA200 (bullish), -1 if EMA50 < EMA200 (bearish)
    trend_12h = np.where(ema50_12h > ema200_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Session filter: 8-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(100, 20)  # Warmup
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(adx[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(trend_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # ADX trend strength filter
        trend_filter = adx[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend weakens OR price moves too far from EMA20
            # Stoploss: 2.5 * ATR(14) below entry (simplified as 2.5% for now)
            if (adx[i] < 20 or  # trend weakening
                close[i] < ema20[i] * 0.975):  # 2.5% below EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend weakens OR price moves too far from EMA20
            if (adx[i] < 20 or  # trend weakening
                close[i] > ema20[i] * 1.025):  # 2.5% above EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pullback to EMA20 in direction of 12h trend
            # Long: price near EMA20 from below, bullish 12h trend, strong ADX
            if (close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                trend_12h_aligned[i] == 1 and
                trend_filter and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price near EMA20 from above, bearish 12h trend, strong ADX
            elif (close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                  close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                  trend_12h_aligned[i] == -1 and
                  trend_filter and
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
6h ADX + Volume Confirmation with 12h Trend Filter
Hypothesis: In trending markets (ADX > 25), price pulls back to the 20-period EMA offer high-probability entries.
The 12h EMA (50/200) filters the trend direction to avoid counter-trend trades. Volume confirms momentum.
Works in bull/bear by only trading with the higher timeframe trend. Target: 80-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_ema_volume_12h_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators ===
    # ADX (14) - trend strength
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([ [np.nan], tr ])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([ [0], dm_plus ])
        dm_minus = np.concatenate([ [0], dm_minus ])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # First average
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(dm_plus_smooth, np.nan)
        minus_di = np.full_like(dm_minus_smooth, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = (atr != 0) & ~np.isnan(atr)
        plus_di[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        minus_di[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx_sum = plus_di + minus_di
        valid_dx = (dx_sum != 0) & ~np.isnan(dx_sum)
        dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / dx_sum[valid_dx]
        
        # ADX - smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            valid_dx_first = ~np.isnan(dx[period:2*period+1])
            if np.any(valid_dx_first):
                first_valid_idx = np.where(valid_dx_first)[0][0] + period
                adx[first_valid_idx] = np.nanmean(dx[first_valid_idx-period+1:first_valid_idx+1])
                
                for i in range(first_valid_idx+1, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # EMA(20) for pullback entries
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema20 = ema(close, 20)
    
    # Volume filter: current volume > 1.5x average of last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) and EMA(200) for trend direction
    ema50_12h = ema(close_12h, 50)
    ema200_12h = ema(close_12h, 200)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Trend: 1 if EMA50 > EMA200 (bullish), -1 if EMA50 < EMA200 (bearish)
    trend_12h = np.where(ema50_12h > ema200_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Session filter: 8-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(100, 20)  # Warmup
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(adx[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(trend_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # ADX trend strength filter
        trend_filter = adx[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend weakens OR price moves too far from EMA20
            # Stoploss: 2.5 * ATR(14) below entry (simplified as 2.5% for now)
            if (adx[i] < 20 or  # trend weakening
                close[i] < ema20[i] * 0.975):  # 2.5% below EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend weakens OR price moves too far from EMA20
            if (adx[i] < 20 or  # trend weakening
                close[i] > ema20[i] * 1.025):  # 2.5% above EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pullback to EMA20 in direction of 12h trend
            # Long: price near EMA20 from below, bullish 12h trend, strong ADX
            if (close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                trend_12h_aligned[i] == 1 and
                trend_filter and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price near EMA20 from above, bearish 12h trend, strong ADX
            elif (close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                  close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                  trend_12h_aligned[i] == -1 and
                  trend_filter and
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
6h ADX + Volume Confirmation with 12h Trend Filter
Hypothesis: In trending markets (ADX > 25), price pulls back to the 20-period EMA offer high-probability entries.
The 12h EMA (50/200) filters the trend direction to avoid counter-trend trades. Volume confirms momentum.
Works in bull/bear by only trading with the higher timeframe trend. Target: 80-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_ema_volume_12h_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators ===
    # ADX (14) - trend strength
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([ [np.nan], tr ])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([ [0], dm_plus ])
        dm_minus = np.concatenate([ [0], dm_minus ])
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # First average
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = np.full_like(dm_plus_smooth, np.nan)
        minus_di = np.full_like(dm_minus_smooth, np.nan)
        dx = np.full_like(atr, np.nan)
        
        valid = (atr != 0) & ~np.isnan(atr)
        plus_di[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        minus_di[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        dx_sum = plus_di + minus_di
        valid_dx = (dx_sum != 0) & ~np.isnan(dx_sum)
        dx[valid_dx] = 100 * np.abs(plus_di[valid_dx] - minus_di[valid_dx]) / dx_sum[valid_dx]
        
        # ADX - smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            valid_dx_first = ~np.isnan(dx[period:2*period+1])
            if np.any(valid_dx_first):
                first_valid_idx = np.where(valid_dx_first)[0][0] + period
                adx[first_valid_idx] = np.nanmean(dx[first_valid_idx-period+1:first_valid_idx+1])
                
                for i in range(first_valid_idx+1, len(dx)):
                    if not np.isnan(dx[i]):
                        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx, plus_di, minus_di
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    
    # EMA(20) for pullback entries
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema20 = ema(close, 20)
    
    # Volume filter: current volume > 1.5x average of last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) and EMA(200) for trend direction
    ema50_12h = ema(close_12h, 50)
    ema200_12h = ema(close_12h, 200)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Trend: 1 if EMA50 > EMA200 (bullish), -1 if EMA50 < EMA200 (bearish)
    trend_12h = np.where(ema50_12h > ema200_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Session filter: 8-20 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(100, 20)  # Warmup
    
    for i in range(start, n):
        # Skip if data not ready
        if np.isnan(adx[i]) or np.isnan(ema20[i]) or np.isnan(vol_ma[i]) or \
           np.isnan(trend_12h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # ADX trend strength filter
        trend_filter = adx[i] > 25
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend weakens OR price moves too far from EMA20
            # Stoploss: 2.5 * ATR(14) below entry (simplified as 2.5% for now)
            if (adx[i] < 20 or  # trend weakening
                close[i] < ema20[i] * 0.975):  # 2.5% below EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend weakens OR price moves too far from EMA20
            if (adx[i] < 20 or  # trend weakening
                close[i] > ema20[i] * 1.025):  # 2.5% above EMA20
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: pullback to EMA20 in direction of 12h trend
            # Long: price near EMA20 from below, bullish 12h trend, strong ADX
            if (close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                trend_12h_aligned[i] == 1 and
                trend_filter and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price near EMA20 from above, bearish 12h trend, strong ADX
            elif (close[i] >= ema20[i] * 0.995 and  # within 0.5% below EMA20
                  close[i] <= ema20[i] * 1.005 and  # within 0.5% above EMA20
                  trend_12h_aligned[i] == -1 and
                  trend_filter and
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
6h ADX + Volume Confirmation with 12h Trend Filter
Hypothesis: In trending markets (ADX > 25), price pulls back to the 20-period EMA offer high-probability entries.
The 12h EMA (50/200) filters the trend direction to avoid counter-trend trades. Volume confirms momentum.
Works in bull/bear by only trading with the higher timeframe trend. Target: 80-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_ema_volume_12h_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 6h Indicators ===
    # ADX (14) - trend strength
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr =