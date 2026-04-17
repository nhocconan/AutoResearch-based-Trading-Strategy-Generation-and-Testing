#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with 1-week EMA34 trend filter and volume confirmation.
# Enters long when price breaks above 20-day high with volume above 1.5x 20-day average and above weekly EMA34.
# Enters short when price breaks below 20-day low with volume above 1.5x 20-day average and below weekly EMA34.
# Exits when price returns to the 10-day moving average or when trend filter fails.
# Designed for low turnover (target: 15-25 trades/year) to minimize fee drag while capturing major trends.
# Works in bull markets (breakout momentum) and bear markets (trend continuation via weekly EMA filter).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day Donchian channels (using previous day's data to avoid look-ahead)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 10-day exit average
    close_10_avg = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().shift(1).values
    
    # Calculate 20-day volume average for confirmation
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    close_10_avg_aligned = align_htf_to_ltf(prices, df_1d, close_10_avg)
    volume_20_avg_aligned = align_htf_to_ltf(prices, df_1d, volume_20_avg)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(close_10_avg_aligned[i]) or 
            np.isnan(volume_20_avg_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_filter = volume[i] > (1.5 * volume_20_avg_aligned[i])
        
        # Trend filter: price relative to weekly EMA34
        price_above_weekly_ema = close[i] > ema34_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume and above weekly EMA34
            if (close[i] > high_20_aligned[i] and volume_filter and price_above_weekly_ema):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume and below weekly EMA34
            elif (close[i] < low_20_aligned[i] and volume_filter and price_below_weekly_ema):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to 10-day average OR breaks below weekly EMA34
            if (close[i] < close_10_avg_aligned[i]) or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to 10-day average OR breaks above weekly EMA34
            if (close[i] > close_10_avg_aligned[i]) or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0