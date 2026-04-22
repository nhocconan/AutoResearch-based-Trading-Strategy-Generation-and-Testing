#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(34) trend filter and volume confirmation
# Targets 25-35 trades per year by requiring strong breakouts in direction of daily trend
# Volume filter ensures breakouts have conviction, trend filter avoids counter-trend trades
# Designed for 12h timeframe to reduce trade frequency and minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period volume moving average for volume confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Reduced threshold for 12h timeframe
    
    # Calculate Donchian channels (20-period) on 12h data
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC (more permissive for 12h)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_aligned[i]) or np.isnan(high_max20[i]) or 
            np.isnan(low_min20[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 20-period high + volume spike + uptrend (close > EMA34)
            if (close[i] > high_max20[i] and vol_spike[i] and close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-period low + volume spike + downtrend (close < EMA34)
            elif (close[i] < low_min20[i] and vol_spike[i] and close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level or trend reversal
            if position == 1:
                # Exit long: price breaks below 20-period low or trend turns down
                if (close[i] < low_min20[i] or close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price breaks above 20-period high or trend turns up
                if (close[i] > high_max20[i] or close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0