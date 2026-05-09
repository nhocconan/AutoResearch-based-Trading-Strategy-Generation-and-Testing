#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels (R3/S3 for exit, R4/S4 for entry)
    R3 = prev_close + 1.1 * prev_range / 6
    S3 = prev_close - 1.1 * prev_range / 6
    R4 = prev_close + 1.1 * prev_range / 2
    S4 = prev_close - 1.1 * prev_range / 2
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get daily trend filter (100 EMA)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: above 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R4_12h[i]) or np.isnan(S4_12h[i]) or 
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 00-23 UTC (all hours for 12h)
        # No session filter needed for 12h timeframe
        
        if position == 0:
            # Long breakout: price breaks above R4 with daily uptrend
            if (close[i] > R4_12h[i] and 
                close[i] > daily_ema_12h[i] and  # daily uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with daily downtrend
            elif (close[i] < S4_12h[i] and 
                  close[i] < daily_ema_12h[i] and  # daily downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R3 (mean reversion)
            if close[i] < R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 (mean reversion)
            if close[i] > S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels (R3/S3 for exit, R4/S4 for entry)
    R3 = prev_close + 1.1 * prev_range / 6
    S3 = prev_close - 1.1 * prev_range / 6
    R4 = prev_close + 1.1 * prev_range / 2
    S4 = prev_close - 1.1 * prev_range / 2
    
    # Align to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R4_12h = align_htf_to_ltf(prices, df_1d, R4)
    S4_12h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Get daily trend filter (100 EMA)
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=100, adjust=False, min_periods=100).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: above 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R4_12h[i]) or np.isnan(S4_12h[i]) or 
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma[i]  # Volume confirmation
        
        if position == 0:
            # Long breakout: price breaks above R4 with daily uptrend
            if (close[i] > R4_12h[i] and 
                close[i] > daily_ema_12h[i] and  # daily uptrend
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S4 with daily downtrend
            elif (close[i] < S4_12h[i] and 
                  close[i] < daily_ema_12h[i] and  # daily downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below R3 (mean reversion)
            if close[i] < R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above S3 (mean reversion)
            if close[i] > S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals