#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1-day trend filter and volume confirmation
# Long when price breaks above 6h 20-period high AND 1d close > 1d 20-period EMA AND volume > 1.5x 20-period average volume
# Short when price breaks below 6h 20-period low AND 1d close < 1d 20-period EMA AND volume > 1.5x 20-period average volume
# Uses 1d trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts.
# Target: 75-150 total trades over 4 years (19-38/year) to stay within optimal range.

name = "6h_donchian20_1d_trend_vol_v1"
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
    
    # Donchian Channel (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Average volume (20-period)
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean()
    
    # 1-day trend filter: EMA(20) on 1d close
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 20-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Align daily EMA to 6h timeframe
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if daily EMA data not available
        if np.isnan(daily_ema_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 6h 20-period low or 1d trend turns bearish
            if (close[i] <= donchian_low[i] or 
                daily_close[i] < daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 6h 20-period high or 1d trend turns bullish
            if (close[i] >= donchian_high[i] or 
                daily_close[i] > daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with 1d trend filter and volume confirmation
            # Long: price breaks above 6h 20-period high AND 1d close > daily EMA AND volume > 1.5x average
            if (close[i] > donchian_high[i] and 
                daily_close[i] > daily_ema_aligned[i] and
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h 20-period low AND 1d close < daily EMA AND volume > 1.5x average
            elif (close[i] < donchian_low[i] and 
                  daily_close[i] < daily_ema_aligned[i] and
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
    
    return signals