#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Enters long when price breaks above Donchian(20) high in bullish 12h trend.
# Enters short when price breaks below Donchian(20) low in bearish 12h trend.
# Volume confirmation: current volume > 1.5x 20-period average volume.
# Designed for 20-50 trades/year on 4h with low drawdown.
# Trend filter from 12h reduces whipsaw in sideways markets.

name = "4h_12h_donchian_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close_12h > ema_12h
    trend_down = close_12h < ema_12h
    
    # Align 12h trend to 4h
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Calculate Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period average volume
    vol_avg_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_avg_20[i]) or \
           np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        long_breakout = close[i] > donch_high[i] and vol_filter and trend_up_aligned[i]
        short_breakout = close[i] < donch_low[i] and vol_filter and trend_down_aligned[i]
        
        # Exit conditions
        exit_long = position == 1 and close[i] < donch_low[i]
        exit_short = position == -1 and close[i] > donch_high[i]
        
        # Priority: entry > exit > hold
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals