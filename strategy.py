#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and 1d trend filter
# Long: price breaks above 20-period high + volume > 1.5x 20-period avg volume + 1d close above 50 EMA
# Short: price breaks below 20-period low + volume > 1.5x 20-period avg volume + 1d close below 50 EMA
# Uses volume filter to avoid false breakouts and 1d trend to align with higher timeframe bias.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "6h_donchian20_12h_vol_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    volume_avg = volume_series.rolling(window=20, min_periods=20).mean()
    
    # 1d trend filter: EMA(50) on daily close
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # 12h volume confirmation: volume ratio
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_12h_avg = volume_12h_series.rolling(window=20, min_periods=20).mean()
    volume_12h_ratio = volume_12h / volume_12h_avg
    volume_12h_ratio_aligned = align_htf_to_ltf(prices, df_12h, volume_12h_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(daily_ema_aligned[i]) or 
            np.isnan(volume_12h_ratio_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 20-period low or 1d trend turns bearish
            if (close[i] <= donchian_low[i] or 
                close[i] < daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-period high or 1d trend turns bullish
            if (close[i] >= donchian_high[i] or 
                close[i] > daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filters
            # Long: price breaks above 20-period high + volume confirmation + bullish 1d trend
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * volume_avg[i] and
                volume_12h_ratio_aligned[i] > 1.2 and
                close[i] > daily_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low + volume confirmation + bearish 1d trend
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * volume_avg[i] and
                  volume_12h_ratio_aligned[i] > 1.2 and
                  close[i] < daily_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals