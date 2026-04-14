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
    
    # Get daily data for monthly pivot points (using month start data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate monthly pivot points using prior month's OHLC
    # Approximate monthly by 21 trading days
    prev_month_high = np.roll(high_1d, 21)
    prev_month_low = np.roll(low_1d, 21)
    prev_month_close = np.roll(close_1d, 21)
    prev_month_high[:21] = np.nan
    prev_month_low[:21] = np.nan
    prev_month_close[:21] = np.nan
    
    # Monthly pivot point
    pp = (prev_month_high + prev_month_low + prev_month_close) / 3
    # Monthly resistance and support levels
    r1 = 2 * pp - prev_month_low
    s1 = 2 * pp - prev_month_high
    r2 = pp + (prev_month_high - prev_month_low)
    s2 = pp - (prev_month_high - prev_month_low)
    
    # Align monthly pivot levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(50) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # for 50-period EMA + 21-day monthly lookback
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above monthly R2 AND above 4h EMA50 with volume confirmation
            if price > r2_aligned[i] and price > ema_50_4h_aligned[i] and vol > 1.3 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below monthly S2 AND below 4h EMA50 with volume confirmation
            elif price < s2_aligned[i] and price < ema_50_4h_aligned[i] and vol > 1.3 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below monthly S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above monthly R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1d_4h_Monthly_Pivot_EMA_Filter"
timeframe = "1h"
leverage = 1.0