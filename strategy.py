#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with 12h EMA trend filter and volume confirmation.
# The Donchian channel provides clear breakout levels from price consolidation. 
# Breakouts above the upper band or below the lower band are filtered by 12h EMA trend direction 
# and confirmed by volume spikes (>1.5x 20-period average). 
# This combination captures strong directional moves while avoiding false breakouts in ranging markets.
# Designed for low trade frequency (target: 20-40 trades/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe (waits for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channel (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average (moderate filter)
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper band + uptrend + volume confirmation
            if price > upper_band and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower band + downtrend + volume confirmation
            elif price < lower_band and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below lower band or trend breaks
                if price < lower_band or price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above upper band or trend breaks
                if price > upper_band or price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0