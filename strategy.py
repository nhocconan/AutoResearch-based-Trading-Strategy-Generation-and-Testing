# [Experiment #20149] 4h_Donchian_20_1d_Trend_Volume_V7
# Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above 20-period high with 1-day EMA uptrend and volume > 1.5x average.
# Short when price breaks below 20-period low with 1-day EMA downtrend and volume > 1.5x average.
# Exit on opposite Donchian break or trend reversal. Designed for 20-50 trades/year to avoid fee drag.
# Works in bull via breakouts, bear via short breakdowns, range via filter.

#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_20_1d_Trend_Volume_V7"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily close for trend
    d_close = df_1d['close'].values
    ema_1d = pd.Series(d_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4-hour Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if daily EMA not available
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Donchian channels not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with uptrend and volume
            long_entry = (close[i] > donchian_high[i]) and (close[i] > ema_1d_aligned[i]) and vol_confirm
            # Short entry: price breaks below Donchian low with downtrend and volume
            short_entry = (close[i] < donchian_low[i]) and (close[i] < ema_1d_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals