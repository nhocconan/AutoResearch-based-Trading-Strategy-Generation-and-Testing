# 12h Donchian Breakout + 1d Trend + Volume Spike
# Hypothesis: Breakouts on 12h with daily trend filter and volume confirmation capture strong moves
# while avoiding false breakouts in chop. Trend filter reduces whipsaw in bear markets like 2022.
# Volume surge confirms institutional participation. Designed for low trade frequency (~15-25/year).

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
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    daily_close = df_1d['close'].values
    daily_ema34 = pd.Series(daily_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Calculate 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current daily close and EMA for trend
        daily_close_val = df_1d['close'].iloc[-1] if len(df_1d) > 0 else np.nan
        daily_ema34_val = daily_ema34_aligned[i]
        
        if np.isnan(daily_close_val) or np.isnan(daily_ema34_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_val > daily_ema34_val
        daily_trend_down = daily_close_val < daily_ema34_val
        
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + daily uptrend + volume spike
            if (high[i] > highest_high[i] and 
                daily_trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + daily downtrend + volume spike
            elif (low[i] < lowest_low[i] and 
                  daily_trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian middle or trend fails
            middle = (highest_high[i] + lowest_low[i]) / 2
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian middle or daily trend turns down
                if low[i] < middle or not daily_trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above Donchian middle or daily trend turns up
                if high[i] > middle or not daily_trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0