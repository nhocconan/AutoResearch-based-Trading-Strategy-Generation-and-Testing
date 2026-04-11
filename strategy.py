# [35144] 1d_1w_donchian_volume_trend_v1
# Hypothesis: Daily Donchian breakout with volume confirmation and weekly trend filter.
# Long when price breaks above Donchian(15) high + volume > 1.8x 20-day average + weekly trend up
# Short when price breaks below Donchian(15) low + volume > 1.8x 20-day average + weekly trend down
# Exit when price returns to Donchian midpoint or weekly trend reverses
# Designed for 10-25 trades/year on 1d timeframe with strong trend capture and low turnover.
# Target: 30-100 total trades over 4 years (7-25/year).
# Weekly trend filter reduces whipsaw in bear markets; volume confirms breakout strength.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(10) for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate 20-day average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (15-period for more sensitivity)
    donchian_high = pd.Series(high).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low).rolling(window=15, min_periods=15).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_10_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.8x 20-day average
        volume_filter = volume[i] > 1.8 * vol_ma_20[i]
        
        # Weekly trend filter: price relative to weekly EMA10
        is_uptrend = close[i] > ema_10_1w_aligned[i]
        is_downtrend = close[i] < ema_10_1w_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and weekly trend
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        donchian_breakdown_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        long_entry = donchian_breakout_up and volume_filter and is_uptrend
        short_entry = donchian_breakdown_down and volume_filter and is_downtrend
        
        # Exit conditions: return to midpoint or weekly trend reversal
        long_exit = (close[i] < donchian_mid[i]) or (not is_uptrend)
        short_exit = (close[i] > donchian_mid[i]) or (not is_downtrend)
        
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