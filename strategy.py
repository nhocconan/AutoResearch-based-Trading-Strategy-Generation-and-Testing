#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h Donchian Breakout with Volume Confirmation and 1d Trend Filter
# Hypothesis: In trending markets, price breaks of 4h Donchian channels (20-period) 
# capture momentum. Volume confirms institutional participation. 1d EMA filter ensures 
# alignment with higher timeframe trend. 1h timeframe provides precise entry timing 
# while 4h/1d filters reduce noise. Designed to work in both bull (breakouts up) 
# and bear (breakouts down) markets with strict entry conditions to limit trades.
name = "1h_donchian_breakout_4h_1d_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (already shifted by 1 for completed bars)
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema_1h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or 
            np.isnan(daily_ema_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 4h Donchian low or trend changes
            if close[i] < donchian_low_1h[i] or close[i] < daily_ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above 4h Donchian high or trend changes
            if close[i] > donchian_high_1h[i] or close[i] > daily_ema_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price breaks above 4h Donchian high with uptrend
                if close[i] > donchian_high_1h[i] and close[i] > daily_ema_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price breaks below 4h Donchian low with downtrend
                elif close[i] < donchian_low_1h[i] and close[i] < daily_ema_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals