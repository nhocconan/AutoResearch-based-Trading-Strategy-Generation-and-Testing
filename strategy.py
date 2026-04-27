#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot levels (calculated from weekly high/low/close) with 1d trend filter and volume confirmation.
# Long when price breaks above weekly R2 with 1d EMA50 uptrend and volume > 1.8x average.
# Short when price breaks below weekly S2 with 1d EMA50 downtrend and volume > 1.8x average.
# Exit when price crosses the weekly pivot point (PP).
# Uses weekly structure for longer-term bias, reducing whipsaw in choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate weekly pivot levels (using previous week's data)
    # PP = (H + L + C) / 3
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    range_weekly = high_weekly - low_weekly
    pp_weekly = (high_weekly + low_weekly + close_weekly) / 3
    r2_weekly = pp_weekly + range_weekly
    s2_weekly = pp_weekly - range_weekly
    
    # Align weekly and daily indicators to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    pp_weekly_aligned = align_htf_to_ltf(prices, df_weekly, pp_weekly)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2_weekly)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2_weekly)
    
    # Get volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA50, weekly pivots, and volume MA20
    start_idx = max(ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(pp_weekly_aligned[i]) or 
            np.isnan(r2_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require significant volume spike
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: break above weekly R2 with 1d EMA50 uptrend and volume spike
            if (price > r2_weekly_aligned[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below weekly S2 with 1d EMA50 downtrend and volume spike
            elif (price < s2_weekly_aligned[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Pivot Point
            if price < pp_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly Pivot Point
            if price > pp_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_Pivot_R2S2_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0