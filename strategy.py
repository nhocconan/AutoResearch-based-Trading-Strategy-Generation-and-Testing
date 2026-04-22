#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20-period) with 1d EMA trend filter and volume confirmation.
# Donchian breakout captures momentum in both bull and bear markets.
# 1d EMA filter ensures we trade with the higher timeframe trend.
# Volume confirmation filters false breakouts.
# Uses 12h timeframe with target of 12-37 trades/year (50-150 total over 4 years).
# Position sizing: 0.25 for long/short to manage drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels on 12h data
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned by get_htf_data)
    # No additional alignment needed as we're using 12h data directly
    
    # Calculate 20-period average volume on 12h for volume confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start loop from sufficient lookback
    for i in range(20, n):
        # Skip if 1d EMA or 12h Donchian data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 12h bar data
        price = close_12h[i]
        dc_high = highest_high[i]
        dc_low = lowest_low[i]
        vol = volume_12h[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirm = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume
            if price > dc_high and price > ema_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume
            elif price < dc_low and price < ema_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or trend reverses
                if price < dc_low or price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or trend reverses
                if price > dc_high or price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0