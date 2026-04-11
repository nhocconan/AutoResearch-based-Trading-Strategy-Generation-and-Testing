# 2025-06-12: 4h 1d/1w strategy using Donchian breakout with volume confirmation and 1d/1w trend filter
# Hypothesis: 4h Donchian breakout with volume > 2x average + 1d/1w trend alignment
# Long when price breaks above Donchian(20) high + volume > 2x average + 1d trend up + 1w trend up
# Short when price breaks below Donchian(20) low + volume > 2x average + 1d trend down + 1w trend down
# Exit when price returns to Donchian midpoint or 1d trend reverses
# Designed for 20-50 trades/year on 4h timeframe with strong trend capture and low turnover
# Targeting BTC/ETH robustness across bull/bear markets

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_donchian_volume_trend_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d and 1w EMA(50) for trend filter
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Trend filters: price relative to 1d and 1w EMA50
        is_1d_uptrend = close[i] > ema_50_1d_aligned[i]
        is_1d_downtrend = close[i] < ema_50_1d_aligned[i]
        is_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        is_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        donchian_breakdown_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        long_entry = donchian_breakout_up and volume_filter and is_1d_uptrend and is_1w_uptrend
        short_entry = donchian_breakdown_down and volume_filter and is_1d_downtrend and is_1w_downtrend
        
        # Exit conditions
        long_exit = (close[i] < donchian_mid[i]) or (not is_1d_uptrend)  # Return to midpoint or 1d trend change
        short_exit = (close[i] > donchian_mid[i]) or (not is_1d_downtrend)  # Return to midpoint or 1d trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals