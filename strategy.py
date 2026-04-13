#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 6h average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # 6h ATR (14-period) for stop-loss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().shift(1).values
    
    # 1d weekly pivot levels (based on previous week)
    # We'll use 1d data to calculate weekly pivot from prior week's OHLC
    # For simplicity, we'll use 1d high/low/close of the previous Friday as weekly pivot
    # Since we don't have direct weekly data in 6s, we'll approximate with 1d
    # Actually, let's use proper 1w data from mtf_data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) > 0:
        # Calculate weekly pivot points from prior week
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Weekly pivot point (PP) = (H + L + C) / 3
        weekly_pp = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1 = (2 * PP) - L
        weekly_r1 = (2 * weekly_pp) - weekly_low
        # Weekly S1 = (2 * PP) - H
        weekly_s1 = (2 * weekly_pp) - weekly_high
        # Weekly R2 = PP + (H - L)
        weekly_r2 = weekly_pp + (weekly_high - weekly_low)
        # Weekly S2 = PP - (H - L)
        weekly_s2 = weekly_pp - (weekly_high - weekly_low)
        
        # Align to 6h timeframe
        weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    else:
        # Fallback if 1w data not available
        weekly_pp_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(avg_vol[i]) or np.isnan(ema_200_1d[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: breakout above upper band + volume confirmation + price above EMA200 + above weekly pivot
            if (price > upper[i] and vol > 2.0 * avg_vol[i] and 
                price > ema_200_1d[i] and price > weekly_pp_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume confirmation + price below EMA200 + below weekly pivot
            elif (price < lower[i] and vol > 2.0 * avg_vol[i] and 
                  price < ema_200_1d[i] and price < weekly_pp_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below EMA200 OR below weekly S1 OR stop-loss hit
            if (price < lower[i] or price < ema_200_1d[i] or 
                price < weekly_s1_aligned[i] or 
                price < entry_price_long - 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above EMA200 OR above weekly R1 OR stop-loss hit
            if (price > upper[i] or price > ema_200_1d[i] or 
                price > weekly_r1_aligned[i] or 
                price > entry_price_short + 2.0 * atr[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Track entry price for stop-loss calculation
        if position != 0 and signals[i] != 0 and (i == start or signals[i-1] == 0):
            if position == 1:
                entry_price_long = close[i]
            else:
                entry_price_short = close[i]
    
    return signals

name = "6h_1w_Donchian_Pivot_Volume_EMA200"
timeframe = "6h"
leverage = 1.0