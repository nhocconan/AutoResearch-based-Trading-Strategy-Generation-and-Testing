#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d/1w Donchian breakout with volume confirmation and 1w trend filter
# Long when price breaks above weekly Donchian(20) high + volume > 1.5x average + weekly trend up
# Short when price breaks below weekly Donchian(20) low + volume > 1.5x average + weekly trend down
# Exit when price returns to weekly Donchian midpoint or trend reverses
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# Weekly trend filter reduces whipsaw in ranging markets while capturing strong trends
# Weekly timeframe aligns with longer-term cycles in BTC/ETH

name = "1d_1w_donchian_volume_trend_v2"
timeframe = "1d"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_1w = (donchian_high_1w + donchian_low_1w) / 2
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    donchian_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_1w)
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to weekly EMA20
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: break of weekly Donchian channel
        donchian_breakout_up = close[i] > donchian_high_1w_aligned[i-1]  # Break above previous week's high
        donchian_breakdown_down = close[i] < donchian_low_1w_aligned[i-1]  # Break below previous week's low
        
        long_entry = donchian_breakout_up and volume_filter and is_uptrend
        short_entry = donchian_breakdown_down and volume_filter and is_downtrend
        
        # Exit conditions: return to midpoint or trend change
        long_exit = (close[i] < donchian_mid_1w_aligned[i]) or (not is_uptrend)
        short_exit = (close[i] > donchian_mid_1w_aligned[i]) or (not is_downtrend)
        
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