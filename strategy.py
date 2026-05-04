#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation (1.5x 20-period EMA)
# Donchian breakouts capture momentum bursts; 12h HMA(21) ensures alignment with higher timeframe trend to avoid counter-trend whipsaws;
# Volume confirmation filters low-conviction breakouts. Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years)
# with discrete sizing (0.25). Works in bull markets by buying breakouts in uptrends and in bear markets by selling breakdowns
# in downtrends, avoiding false breakouts in ranging markets via volume and trend filters.

name = "4h_Donchian20_12hHMA21_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend filter
    close_12h = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = pd.Series(close_12h).ewm(span=half_len, adjust=False, min_periods=half_len).mean().values
    wma_full = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_12h = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False, min_periods=sqrt_len).mean().values
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate Donchian(20) channels on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.5x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start from 20 to have valid Donchian
        # Skip if any value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume confirmation + price above 12h HMA21 (uptrend)
            if (close[i] > highest_high[i] and volume_confirmed and 
                close[i] > hma_21_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume confirmation + price below 12h HMA21 (downtrend)
            elif (close[i] < lowest_low[i] and volume_confirmed and 
                  close[i] < hma_21_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR price below 12h HMA21 (trend change)
            if close[i] < lowest_low[i] or close[i] < hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR price above 12h HMA21 (trend change)
            if close[i] > highest_high[i] or close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals