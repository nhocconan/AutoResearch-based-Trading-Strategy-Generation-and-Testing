#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w trend filter
# Long when price breaks above 4h 20-bar high AND daily volume > 1.5x 20-bar avg AND weekly close > weekly EMA(20)
# Short when price breaks below 4h 20-bar low AND daily volume > 1.5x 20-bar avg AND weekly close < weekly EMA(20)
# Uses volume to confirm breakout strength and weekly trend to avoid counter-trend trades.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within optimal range.

name = "4h_donchian20_1d_vol_1w_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period average volume
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean()
    
    # Weekly trend filter: EMA(20) on weekly close
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema = weekly_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily volume filter (for confirmation)
    df_1d = get_htf_data(prices, '1d')
    daily_volume = df_1d['volume'].values
    daily_volume_series = pd.Series(daily_volume)
    daily_vol_avg = daily_volume_series.rolling(window=20, min_periods=20).mean()
    daily_vol_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_vol_avg.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(weekly_ema_aligned[i]) or 
            np.isnan(daily_vol_avg_aligned[i]) or
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        daily_vol_confirm = daily_volume[i] > 1.5 * daily_vol_avg_aligned[i]
        
        # Check exits
        if position == 1:  # long position
            # Exit: price breaks below 20-bar low or weekly trend turns bearish
            if (close[i] <= donchian_low[i] or 
                weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 20-bar high or weekly trend turns bullish
            if (close[i] >= donchian_high[i] or 
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and weekly trend filter
            # Long: price breaks above 20-bar high AND volume confirms AND weekly close above weekly EMA
            if (close[i] > donchian_high[i] and 
                vol_confirm and daily_vol_confirm and
                weekly_close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-bar low AND volume confirms AND weekly close below weekly EMA
            elif (close[i] < donchian_low[i] and 
                  vol_confirm and daily_vol_confirm and
                  weekly_close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
    
    return signals