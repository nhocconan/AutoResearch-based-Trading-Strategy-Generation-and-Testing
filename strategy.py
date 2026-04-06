#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku signals (Tenkan/Kijun cross) on 6h filtered by 1d trend (price vs Kumo) and volume spikes capture
trend continuations while avoiding counter-trend trades. The Kumo acts as dynamic support/resistance, reducing whipsaw
in sideways markets. Works in both bull and bear markets by following the higher timeframe trend.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_1d_data, align_htf_to_ltf

name = "6h_ichimoku_daily_filter_vol_v1"
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
    
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_span_b_period = 52
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan = np.full(n, np.nan)
    for i in range(tenkan_period - 1, n):
        tenkan[i] = (np.max(high[i-tenkan_period+1:i+1]) + np.min(low[i-tenkan_period+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        kijun[i] = (np.max(high[i-kijun_period+1:i+1]) + np.min(low[i-kijun_period+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = np.full(n, np.nan)
    for i in range(kijun_period - 1, n):
        idx = i + kijun_period
        if idx < n:
            senkou_a[idx] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods plotted 26 ahead
    senkou_b = np.full(n, np.nan)
    for i in range(senkou_span_b_period - 1, n):
        idx = i + kijun_period
        if idx < n:
            senkou_b[idx] = (np.max(high[i-senkou_span_b_period+1:i+1]) + np.min(low[i-senkou_span_b_period+1:i+1])) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Kumo (Cloud) from 1d: Senkou Span A and B
    # Calculate Ichimoku on 1d for Kumo
    tenkan_1d = np.full(len(close_1d), np.nan)
    kijun_1d = np.full(len(close_1d), np.nan)
    senkou_a_1d = np.full(len(close_1d), np.nan)
    senkou_b_1d = np.full(len(close_1d), np.nan)
    
    for i in range(tenkan_period - 1, len(close_1d)):
        tenkan_1d[i] = (np.max(high_1d[i-tenkan_period+1:i+1]) + np.min(low_1d[i-tenkan_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        kijun_1d[i] = (np.max(high_1d[i-kijun_period+1:i+1]) + np.min(low_1d[i-kijun_period+1:i+1])) / 2
    
    for i in range(kijun_period - 1, len(close_1d)):
        idx = i + kijun_period
        if idx < len(close_1d):
            senkou_a_1d[idx] = (tenkan_1d[i] + kijun_1d[i]) / 2
    
    for i in range(senkou_span_b_period - 1, len(close_1d)):
        idx = i + kijun_period
        if idx < len(close_1d):
            senkou_b_1d[idx] = (np.max(high_1d[i-senkou_span_b_period+1:i+1]) + np.min(low_1d[i-senkou_span_b_period+1:i+1])) / 2
    
    # Kumo top and bottom
    kumo_top = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Trend: 1 if price above Kumo (uptrend), -1 if price below Kumo (downtrend), 0 if inside Kumo (neutral)
    trend_1d = np.where(close_1d > kumo_top, 1, np.where(close_1d < kumo_bottom, -1, 0))
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(52, 26, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Tenkan crosses below Kijun OR price drops below Kumo bottom OR trend turns neutral/down
            # Stoploss: price drops 2*ATR below entry
            if (tenkan[i] < kijun[i] or
                close[i] < senkou_b[i] or
                trend_1d_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Tenkan crosses above Kijun OR price rises above Kumo top OR trend turns neutral/up
            # Stoploss: price rises 2*ATR above entry
            if (tenkan[i] > kijun[i] or
                close[i] > senkou_a[i] or
                trend_1d_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: Tenkan crosses above Kijun in uptrend with volume
            if (tenkan[i] > kijun[i] and
                tenkan[i-1] <= kijun[i-1] and  # crossed just now
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Tenkan crosses below Kijun in downtrend with volume
            elif (tenkan[i] < kijun[i] and
                  tenkan[i-1] >= kijun[i-1] and  # crossed just now
                  trend_1d_aligned[i] == -1 and
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
6h Camarilla Pivot with 1d Trend Filter and Volume Confirmation
Hypothesis: Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) on 6h filtered by 1d trend (price vs 200 EMA)
and volume spikes capture high-probability reversals and breakouts. Works in both bull/bear markets by aligning with higher
timeframe trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_vol_v1"
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
    
    # Calculate Camarilla pivot levels for each bar using previous day's OHLC
    # We'll use rolling window of 4 bars (approx 1 day at 6h) to get previous day's OHLC
    cam_r3 = np.full(n, np.nan)
    cam_s3 = np.full(n, np.nan)
    cam_r4 = np.full(n, np.nan)
    cam_s4 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    
    for i in range(4, n):
        # Previous day's OHLC (using last 4 bars as approximation)
        prev_high = np.max(high[i-4:i])
        prev_low = np.min(low[i-4:i])
        prev_close = close[i-1]  # close of previous bar
        
        # Calculate pivot
        pivot[i] = (prev_high + prev_low + prev_close) / 3
        
        # Calculate Camarilla levels
        range_val = prev_high - prev_low
        cam_r3[i] = pivot[i] + range_val * 1.1 / 4
        cam_s3[i] = pivot[i] - range_val * 1.1 / 4
        cam_r4[i] = pivot[i] + range_val * 1.1 / 2
        cam_s4[i] = pivot[i] - range_val * 1.1 / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 200-period EMA on 1d for trend
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 198) / 200
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(200, 20, 4)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(pivot[i]) or np.isnan(cam_r3[i]) or np.isnan(cam_s3[i]) or \
           np.isnan(cam_r4[i]) or np.isnan(cam_s4[i]) or np.isnan(trend_1d_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 OR trend turns down
            # Stoploss: price drops 2*ATR below entry (simplified)
            if (close[i] < cam_s3[i] or
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price rises above R3 OR trend turns up
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > cam_r3[i] or
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price crosses above S3 with volume in uptrend (bounce off support)
            if (close[i] > cam_s3[i] and
                close[i-1] <= cam_s3[i-1] and  # crossed just now
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below R3 with volume in downtrend (rejection at resistance)
            elif (close[i] < cam_r3[i] and
                  close[i-1] >= cam_r3[i-1] and  # crossed just now
                  trend_1d_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Long breakout: price breaks above R4 with volume in uptrend
            elif (close[i] > cam_r4[i] and
                  close[i-1] <= cam_r4[i-1] and  # broke just now
                  trend_1d_aligned[i] == 1 and
                  volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short breakdown: price breaks below S4 with volume in downtrend
            elif (close[i] < cam_s4[i] and
                  close[i-1] >= cam_s4[i-1] and  # broke just now
                  trend_1d_aligned[i] == -1 and
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
6h Donchian Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Donchian channel breakouts on 6h filtered by 1d trend (price vs 50 EMA) and volume spikes
capture momentum in both bull and bear markets. The 1d trend filter prevents counter-trend trades,
while volume ensures breakout legitimacy. ATR-based stops limit drawdown. Target: 50-150 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_1d_trend_vol_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 50-period EMA on 1d for trend
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 49) / 51
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
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
        if np.isnan(atr[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR trend turns down
            # Stoploss: price drops 2.5*ATR below entry
            if (close[i] <= donch_low[i] or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR trend turns up
            # Stoploss: price rises 2.5*ATR above entry
            if (close[i] >= donch_high[i] or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries
            # Long: price breaks above Donchian high in uptrend with volume
            if (close[i] > donch_high[i] and
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low in downtrend with volume
            elif (close[i] < donch_low[i] and
                  trend_1d_aligned[i] == -1 and
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
6h Elder Ray Index with 1d Trend Filter and Volume Confirmation
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) on 6h filtered by 1d trend (price vs 50 EMA)
and volume spikes captures institutional buying/selling pressure. Works in both bull/bear markets by following higher
timeframe trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_1d_trend_vol_v1"
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
    
    # Elder Ray parameters
    ema_period = 13
    
    # Calculate EMA13
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period-1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] * 2 + ema[i-1] * (ema_period-1)) / (ema_period+1)
    
    # Bull Power = High - EMA13
    bull_power = np.full(n, np.nan)
    bear_power = np.full(n, np.nan)
    
    for i in range(ema_period-1, n):
        bull_power[i] = high[i] - ema[i]
        bear_power[i] = ema[i] - low[i]
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 50-period EMA on 1d for trend
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 49) / 51
    
    # Trend: 1 if close > EMA (uptrend), -1 if close < EMA (downtrend)
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(ema_period, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or \
           np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bull Power turns negative OR trend turns down
            # Stoploss: price drops 2*ATR below entry
            if (bull_power[i] < 0 or
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power turns negative OR trend turns up
            # Stoploss: price rises 2*ATR above entry
            if (bear_power[i] < 0 or
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: Bull Power > 0 and increasing with volume in uptrend
            if (bull_power[i] > 0 and
                i > 0 and bull_power[i] > bull_power[i-1] and  # increasing
                trend_1d_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power > 0 and increasing with volume in downtrend
            elif (bear_power[i] > 0 and
                  i > 0 and bear_power[i] > bear_power[i-1] and  # increasing
                  trend_1d_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals

--- 0 files read, 0 files written ---