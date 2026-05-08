#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d close for Camarilla calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d ATR(14) for volatility normalization (used in Camarilla)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 12h ATR(14) for volume spike filter
    tr_12h = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume moving average for spike detection
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr14_12h[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        # Camarilla levels are based on previous day's range
        prev_idx = i - 1
        if prev_idx < 0:
            continue
            
        # Get previous day's OHLC (need to find the actual 1d bar)
        # Since we're on 12h timeframe, we need to check if we have crossed a day boundary
        # Simplified: use rolling window of 2 periods (24h) to approximate daily OHLC
        if i >= 2:
            # Approximate daily OHLC using last 2 periods (24h of 12h data)
            day_high = np.max(high[i-1:i+1])  # current and previous period
            day_low = np.min(low[i-1:i+1])
            day_close = close[i-1]  # previous period close
        else:
            continue
            
        # Calculate Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            continue
            
        # Camarilla levels
        R4 = day_close + range_val * 1.1 / 2
        R3 = day_close + range_val * 1.1 / 4
        R2 = day_close + range_val * 1.1 / 6
        R1 = day_close + range_val * 1.1 / 12
        S1 = day_close - range_val * 1.1 / 12
        S2 = day_close - range_val * 1.1 / 6
        S3 = day_close - range_val * 1.1 / 4
        S4 = day_close - range_val * 1.1 / 2
        
        ema_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and above weekly EMA
            if (close[i] > R3 and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and below weekly EMA
            elif (close[i] < S3 and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or below weekly EMA
            if (close[i] < S3 or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or above weekly EMA
            if (close[i] > R3 or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d close for Camarilla calculation
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 1d ATR(14) for volatility normalization (used in Camarilla)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    atr14_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 12h ATR(14) for volume spike filter
    tr_12h = np.maximum(high[1:] - low[1:], 
                        np.maximum(np.abs(high[1:] - close[:-1]), 
                                   np.abs(low[1:] - close[:-1])))
    tr_12h = np.concatenate([[np.nan], tr_12h])
    atr14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume moving average for spike detection
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr14_12h[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        # Camarilla levels are based on previous day's range
        prev_idx = i - 1
        if prev_idx < 0:
            continue
            
        # Get previous day's OHLC (need to find the actual 1d bar)
        # Since we're on 12h timeframe, we need to check if we have crossed a day boundary
        # Simplified: use rolling window of 2 periods (24h) to approximate daily OHLC
        if i >= 2:
            # Approximate daily OHLC using last 2 periods (24h of 12h data)
            day_high = np.max(high[i-1:i+1])  # current and previous period
            day_low = np.min(low[i-1:i+1])
            day_close = close[i-1]  # previous period close
        else:
            continue
            
        # Calculate Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            continue
            
        # Camarilla levels
        R4 = day_close + range_val * 1.1 / 2
        R3 = day_close + range_val * 1.1 / 4
        R2 = day_close + range_val * 1.1 / 6
        R1 = day_close + range_val * 1.1 / 12
        S1 = day_close - range_val * 1.1 / 12
        S2 = day_close - range_val * 1.1 / 6
        S3 = day_close - range_val * 1.1 / 4
        S4 = day_close - range_val * 1.1 / 2
        
        ema_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike and above weekly EMA
            if (close[i] > R3 and vol_spike and close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike and below weekly EMA
            elif (close[i] < S3 and vol_spike and close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or below weekly EMA
            if (close[i] < S3 or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or above weekly EMA
            if (close[i] > R3 or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals