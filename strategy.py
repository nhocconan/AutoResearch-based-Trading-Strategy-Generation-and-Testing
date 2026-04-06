#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day trend filter and volume confirmation
# Long when price breaks above 4h 20-period high AND 1d close > 1d EMA(50) AND volume > 1.5x 20-period avg volume
# Short when price breaks below 4h 20-period low AND 1d close < 1d EMA(50) AND volume > 1.5x 20-period avg volume
# Uses 1-day trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Average volume for confirmation
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=20, min_periods=20).mean()
    
    # 1-day trend filter: EMA(50) on daily close
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close
    daily_close_series = pd.Series(daily_close)
    daily_ema = daily_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily EMA to 4h timeframe
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
            # Exit: price breaks below 4h 20-period low OR daily trend turns bearish
            if (close[i] <= donchian_low[i] or 
                daily_close[i] < daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 4h 20-period high OR daily trend turns bullish
            if (close[i] >= donchian_high[i] or 
                daily_close[i] > daily_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above 4h 20-period high AND daily close above daily EMA AND volume > 1.5x avg
            vol_condition = volume[i] > 1.5 * avg_volume[i] if not np.isnan(avg_volume[i]) else False
            if (close[i] > donchian_high[i] and 
                daily_close[i] > daily_ema_aligned[i] and
                vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4h 20-period low AND daily close below daily EMA AND volume > 1.5x avg
            elif (close[i] < donchian_low[i] and 
                  daily_close[i] < daily_ema_aligned[i] and
                  vol_condition):
                signals[i] = -0.25
                position = -1
    
    return signals