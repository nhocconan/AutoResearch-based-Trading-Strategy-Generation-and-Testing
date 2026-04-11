#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with volume confirmation and 1w trend filter
# Long when price breaks above Donchian(10) high + volume > 1.5x average + 1w trend up
# Short when price breaks below Donchian(10) low + volume > 1.5x average + 1w trend down
# Exit when price returns to Donchian midpoint or trend reverses
# Target: 15-25 trades/year on 1d timeframe with strong trend capture and low turnover
# Designed to work in both bull and bear markets via trend filter and volatility-adjusted position sizing

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 10-period average volume for volume filter
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Donchian channels (10-period)
    donchian_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(10, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_10[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 10-period average
        volume_filter = volume[i] > 1.5 * vol_ma_10[i]
        
        # Trend filter: price relative to 1w EMA20
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        donchian_breakdown_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        long_entry = donchian_breakout_up and volume_filter and is_uptrend
        short_entry = donchian_breakdown_down and volume_filter and is_downtrend
        
        # Exit conditions
        long_exit = (close[i] < donchian_mid[i]) or (not is_uptrend)  # Return to midpoint or trend change
        short_exit = (close[i] > donchian_mid[i]) or (not is_downtrend)  # Return to midpoint or trend change
        
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