#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data
    # We need to map 4h bars to their corresponding previous day
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # For each 4h bar, use previous day's OHLC (we'll approximate by using 6 periods back for 4h data)
    # Since 1 day = 6 * 4h bars, we use shift(6) to get previous day's levels
    if len(close_4h) >= 7:
        # Calculate pivot and ranges using previous day's data (6 bars back)
        high_prev = np.roll(high_4h, 6)
        low_prev = np.roll(low_4h, 6)
        close_prev = np.roll(close_4h, 6)
        
        # For first 6 bars, we don't have previous day data
        high_prev[:6] = np.nan
        low_prev[:6] = np.nan
        close_prev[:6] = np.nan
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        camarilla_r3 = pivot + (range_val * 1.1000 / 4.0)  # R3 level
        camarilla_s3 = pivot - (range_val * 1.1000 / 4.0)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume filter
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume filter
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data
    # We need to map 4h bars to their corresponding previous day
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # For each 4h bar, use previous day's OHLC (we'll approximate by using 6 periods back for 4h data)
    # Since 1 day = 6 * 4h bars, we use shift(6) to get previous day's levels
    if len(close_4h) >= 7:
        # Calculate pivot and ranges using previous day's data (6 bars back)
        high_prev = np.roll(high_4h, 6)
        low_prev = np.roll(low_4h, 6)
        close_prev = np.roll(close_4h, 6)
        
        # For first 6 bars, we don't have previous day data
        high_prev[:6] = np.nan
        low_prev[:6] = np.nan
        close_prev[:6] = np.nan
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        camarilla_r3 = pivot + (range_val * 1.1000 / 4.0)  # R3 level
        camarilla_s3 = pivot - (range_val * 1.1000 / 4.0)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume filter
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume filter
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data
    # We need to map 4h bars to their corresponding previous day
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # For each 4h bar, use previous day's OHLC (we'll approximate by using 6 periods back for 4h data)
    # Since 1 day = 6 * 4h bars, we use shift(6) to get previous day's levels
    if len(close_4h) >= 7:
        # Calculate pivot and ranges using previous day's data (6 bars back)
        high_prev = np.roll(high_4h, 6)
        low_prev = np.roll(low_4h, 6)
        close_prev = np.roll(close_4h, 6)
        
        # For first 6 bars, we don't have previous day data
        high_prev[:6] = np.nan
        low_prev[:6] = np.nan
        close_prev[:6] = np.nan
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        camarilla_r3 = pivot + (range_val * 1.1000 / 4.0)  # R3 level
        camarilla_s3 = pivot - (range_val * 1.1000 / 4.0)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume filter
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume filter
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data
    # We need to map 4h bars to their corresponding previous day
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # For each 4h bar, use previous day's OHLC (we'll approximate by using 6 periods back for 4h data)
    # Since 1 day = 6 * 4h bars, we use shift(6) to get previous day's levels
    if len(close_4h) >= 7:
        # Calculate pivot and ranges using previous day's data (6 bars back)
        high_prev = np.roll(high_4h, 6)
        low_prev = np.roll(low_4h, 6)
        close_prev = np.roll(close_4h, 6)
        
        # For first 6 bars, we don't have previous day data
        high_prev[:6] = np.nan
        low_prev[:6] = np.nan
        close_prev[:6] = np.nan
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        camarilla_r3 = pivot + (range_val * 1.1000 / 4.0)  # R3 level
        camarilla_s3 = pivot - (range_val * 1.1000 / 4.0)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume filter
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume filter
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data
    # We need to map 4h bars to their corresponding previous day
    camarilla_r3 = np.full(len(close_4h), np.nan)
    camarilla_s3 = np.full(len(close_4h), np.nan)
    
    # For each 4h bar, use previous day's OHLC (we'll approximate by using 6 periods back for 4h data)
    # Since 1 day = 6 * 4h bars, we use shift(6) to get previous day's levels
    if len(close_4h) >= 7:
        # Calculate pivot and ranges using previous day's data (6 bars back)
        high_prev = np.roll(high_4h, 6)
        low_prev = np.roll(low_4h, 6)
        close_prev = np.roll(close_4h, 6)
        
        # For first 6 bars, we don't have previous day data
        high_prev[:6] = np.nan
        low_prev[:6] = np.nan
        close_prev[:6] = np.nan
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_val = high_prev - low_prev
        
        camarilla_r3 = pivot + (range_val * 1.1000 / 4.0)  # R3 level
        camarilla_s3 = pivot - (range_val * 1.1000 / 4.0)  # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma34_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above Camarilla R3 + daily uptrend + volume filter
            if close[i] > camarilla_r3_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 + daily downtrend + volume filter
            elif close[i] < camarilla_s3_aligned[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below Camarilla S3 or daily trend down
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above Camarilla R3 or daily trend up
            if close[i] > camarilla_r3_aligned[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
name = "4h_4H_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: close above/below 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # Daily volume filter: volume > 1.5x 34-day average
    vol_1d = df_1d['volume'].values
    vol_ma34_1d = pd.Series(vol_1d).rolling(window=34, min_periods=34).mean().values
    vol_ma34_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma34_1d)
    volume_filter = volume > 1.5 * vol_ma34_1d_aligned
    
    # 4h Camarilla levels from previous day (use previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla R3, S3 levels for each 4h bar using previous day's data