#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with 1-week EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + price > 1w EMA34 + volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low + price < 1w EMA34 + volume > 1.5x 20-period average
# Exit when price crosses back through Donchian midpoint or EMA trend reverses
# Donchian provides clear breakout levels, EMA34 filters trend direction, volume confirms conviction
# Target: 15-30 trades/year by requiring breakout + trend alignment + volume spike

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h timeframe
    ema34_12h = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Calculate Donchian channels (20-period) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Calculate 12h volume moving average (20-period)
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        ema34_val = ema34_12h[i]
        vol_ma_val = vol_ma[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Price breaks above Donchian high + above EMA34 + volume confirmation
            if price > donch_high_val and price > ema34_val and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below EMA34 + volume confirmation
            elif price < donch_low_val and price < ema34_val and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price crosses below Donchian midpoint or below EMA34
                if price < donch_mid_val or price < ema34_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price crosses above Donchian midpoint or above EMA34
                if price > donch_mid_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0