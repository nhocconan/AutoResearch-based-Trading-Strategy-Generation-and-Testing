#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
    # Long when price breaks above Donchian upper (20) with EMA12h > EMA26_12h and volume > 1.5x average.
    # Short when price breaks below Donchian lower (20) with EMA12h < EMA26_12h and volume > 1.5x average.
    # Exit when price touches Donchian midpoint.
    # Uses structure breakouts in trending markets to capture momentum, avoids false breakouts in ranging markets.
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA(12) and EMA(26) on 12h for trend filter
    ema_12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26_12h = pd.Series(close_12h).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12_12h)
    ema_26_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_26_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_12_12h_aligned[i]) or np.isnan(ema_26_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA12 > EMA26 for uptrend, EMA12 < EMA26 for downtrend
        uptrend = ema_12_12h_aligned[i] > ema_26_12h_aligned[i]
        downtrend = ema_12_12h_aligned[i] < ema_26_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = (high[i] > donchian_upper[i]) and uptrend and volume_confirm
        short_breakout = (low[i] < donchian_lower[i]) and downtrend and volume_confirm
        
        # Exit conditions: price touches Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_ema_volume_v1"
timeframe = "4h"
leverage = 1.0