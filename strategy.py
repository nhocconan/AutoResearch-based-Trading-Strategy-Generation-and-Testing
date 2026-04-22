#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high + price > 1d EMA34 + volume > 1.5x 20-period average.
# Short when price breaks below Donchian(20) low + price < 1d EMA34 + volume > 1.5x 20-period average.
# Exit when price crosses 1d EMA34 (trend reversal) or after 3 bars (time-based exit to avoid whipsaw).
# Designed for low trade frequency (~15-30/year) to minimize fee drift. Works in bull/bear markets
# by using Donchian breakouts for momentum and EMA filter to avoid counter-trend trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period high/low) on 12h close
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_34_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions: trend reversal or time-based exit
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below EMA or after 3 bars
                if price < ema_val or bars_since_entry >= 3:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above EMA or after 3 bars
                if price > ema_val or bars_since_entry >= 3:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0