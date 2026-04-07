#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: On 6-hour timeframe, use daily Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) 
with EMA(50) trend filter and volume confirmation. Fade at R3/S3 when price rejects extreme with 
contrarian signal, breakout at R4/S4 when price breaks with trend alignment. Designed for 15-30 
trades/year to minimize fee drift while capturing mean reversion in ranges and breakouts in trends.
Works in both bull/bear markets as Camarilla levels adapt to volatility and EMA filter avoids 
counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Determine daily trend direction (using EMA slope)
    daily_trend_up = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    daily_trend_down = np.zeros(len(ema_50_1d_aligned), dtype=bool)
    for i in range(1, len(ema_50_1d_aligned)):
        if not np.isnan(ema_50_1d_aligned[i]) and not np.isnan(ema_50_1d_aligned[i-1]):
            daily_trend_up[i] = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            daily_trend_down[i] = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pp = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_ * 1.500
    r3 = close_1d + range_ * 1.250
    s3 = close_1d - range_ * 1.250
    s4 = close_1d - range_ * 1.500
    
    # Align all levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume filter: 20-period average on 6h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (mean reversion target) or breaks S4 with trend
            if close[i] <= s3_aligned[i] or (close[i] < s4_aligned[i] and daily_trend_down[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R3 (mean reversion target) or breaks R4 with trend
            if close[i] >= r3_aligned[i] or (close[i] > r4_aligned[i] and daily_trend_up[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Fade at R3/S3: price rejects extreme level
                # Long fade: price rejects S3 with rejection candle
                if (close[i] > s3_aligned[i] and open_price[i] <= s3_aligned[i] and 
                    close[i] >= open_price[i] and daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short fade: price rejects R3 with rejection candle
                elif (close[i] < r3_aligned[i] and open_price[i] >= r3_aligned[i] and 
                      close[i] <= open_price[i] and daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
                # Breakout at R4/S4: price breaks extreme with trend alignment
                # Long breakout: price breaks R4 with bullish candle and uptrend
                elif (close[i] > r4_aligned[i] and open_price[i] <= r4_aligned[i] and 
                      close[i] > open_price[i] and daily_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks S4 with bearish candle and downtrend
                elif (close[i] < s4_aligned[i] and open_price[i] >= s4_aligned[i] and 
                      close[i] < open_price[i] and daily_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals