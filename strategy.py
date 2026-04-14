#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(34) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    prev_day_high = np.roll(high_1d, 1)
    prev_day_low = np.roll(low_1d, 1)
    prev_day_close = np.roll(close_1d, 1)
    prev_day_high[0] = np.nan
    prev_day_low[0] = np.nan
    prev_day_close[0] = np.nan
    
    # Daily pivot point
    pp = (prev_day_high + prev_day_low + prev_day_close) / 3
    # Daily resistance and support levels
    r2 = pp + (prev_day_high - prev_day_low)
    s2 = pp - (prev_day_high - prev_day_low)
    
    # Align daily pivot levels to 1d timeframe (using prior day's data)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # for 34-period EMA and 20-period volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above daily R2 AND above weekly EMA34 with volume filter
            if (price > r2_aligned[i] and price > ema_34_1w_aligned[i] and 
                vol > 1.8 * avg_vol[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below daily S2 AND below weekly EMA34 with volume filter
            elif (price < s2_aligned[i] and price < ema_34_1w_aligned[i] and 
                  vol > 1.8 * avg_vol[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below daily pivot point (PP)
            # Recalculate PP for current day's exit
            pp_current = (high_1d[i] + low_1d[i] + close_1d[i // 1]) / 3 if i > 0 else np.nan
            if i > 0 and not np.isnan(pp_current) and price < pp_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above daily pivot point (PP)
            pp_current = (high_1d[i] + low_1d[i] + close_1d[i // 1]) / 3 if i > 0 else np.nan
            if i > 0 and not np.isnan(pp_current) and price > pp_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Daily_Pivot_Weekly_EMA_Volume_Filter"
timeframe = "1d"
leverage = 1.0