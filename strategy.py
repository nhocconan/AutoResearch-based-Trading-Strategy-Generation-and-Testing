# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h/1d trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 12h EMA50 rising AND volume spike
# Short when price breaks below Donchian(20) low AND 12h EMA50 falling AND volume spike
# Exit when price crosses back through Donchian(20) midline OR volume dries up
# Uses Donchian channels for breakout signals, 12h EMA50 for trend filter, volume spike for confirmation
# Designed for low trade frequency (~15-30/year) with edge in trending markets
# Works in both bull (breakouts up) and bear (breakdowns down) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_50_rising = ema_50_12h > ema_50_12h_prev
    ema_50_falling = ema_50_12h < ema_50_12h_prev
    
    # Align 12h EMA trend to 6t
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Calculate Donchian channels (20-period) on 6t data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Donchian high (20-period rolling max)
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian low (20-period rolling min)
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Donchian midline (average of high and low)
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema_rising = ema_50_rising_aligned[i]
        ema_falling = ema_50_falling_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 12h EMA50 rising AND volume spike
            if price > donch_high_val and ema_rising and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 12h EMA50 falling AND volume spike
            elif price < donch_low_val and ema_falling and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through Donchian midline OR volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price falls below midline or volume dries up
                if price < donch_mid_val or vol < vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above midline or volume dries up
                if price > donch_mid_val or vol < vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_Breakout_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0