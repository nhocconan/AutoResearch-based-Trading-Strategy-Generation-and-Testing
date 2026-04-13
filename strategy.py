#!/usr/bin/env python3
"""
1h_4h_1d_Three_Timeframe_Trend_Follow
Hypothesis: In trending markets, price respects higher timeframe EMAs.
The strategy uses 4h EMA20 for medium trend and 1d EMA50 for long trend alignment.
Entries occur on 1h when price pulls back to the 4h EMA20 in the direction of the 1d trend.
Exit when price crosses the 4h EMA20 against the trend.
This captures trend continuation with defined risk and avoids choppy markets.
Target: 20-40 trades/year by requiring alignment across three timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA20 (medium-term trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for EMA50 (long-term trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h EMA20 for entry timing and exit
    close_series = pd.Series(close)
    ema_20_1h = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Volume filter: avoid low volume periods
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > (volume_ma_20 * 0.5)  # at least half of average volume
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_20_1h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside active session (08:00-20:00 UTC)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Determine trend direction from higher timeframes
        # Long trend: price above both 4h EMA20 and 1d EMA50
        # Short trend: price below both 4h EMA20 and 1d EMA50
        long_trend = (close[i] > ema_20_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        short_trend = (close[i] < ema_20_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        # Entry conditions: pullback to 4h EMA20 in direction of trend
        long_entry = long_trend and (close[i] <= ema_20_4h_aligned[i] * 1.002) and (close[i] >= ema_20_4h_aligned[i] * 0.998)
        short_entry = short_trend and (close[i] >= ema_20_4h_aligned[i] * 0.998) and (close[i] <= ema_20_4h_aligned[i] * 1.002)
        
        # Exit conditions: price crosses 4h EMA20 against the trend
        long_exit = not long_trend and (close[i] < ema_20_4h_aligned[i])
        short_exit = not short_trend and (close[i] > ema_20_4h_aligned[i])
        
        if (long_entry or short_entry) and volume_filter[i]:
            if long_entry and position != 1:
                position = 1
                signals[i] = position_size
            elif short_entry and position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0  # no change
        elif long_exit and position == 1:
            position = 0
            signals[i] = 0.0
        elif short_exit and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_1d_Three_Timeframe_Trend_Follow"
timeframe = "1h"
leverage = 1.0