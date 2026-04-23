#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1w EMA34 AND volume > 2.0x average.
Short when price breaks below Donchian(20) low AND price < 1w EMA34 AND volume > 2.0x average.
Exit when price crosses 1w EMA34 or volume drops below average.
Donchian breakouts capture strong momentum; 1w EMA34 ensures alignment with weekly trend.
Volume confirmation avoids false breakouts. Designed for 1d timeframe targeting 30-100 total trades over 4 years.
Works in both bull and bear markets by only taking trades aligned with 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian channels to 1d timeframe (no shift needed as we're already on 1d)
    donchian_high_aligned = donchian_high  # Already aligned to 1d
    donchian_low_aligned = donchian_low    # Already aligned to 1d
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high = donchian_high_aligned[i]
        donch_low = donchian_low_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND price > 1w EMA34 AND volume spike
            if (price > donch_high and price > ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND price < 1w EMA34 AND volume spike
            elif (price < donch_low and price < ema34_val and vol_current > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 1w EMA34 OR volume drops below average
                if (price < ema34_val or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 1w EMA34 OR volume drops below average
                if (price > ema34_val or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0