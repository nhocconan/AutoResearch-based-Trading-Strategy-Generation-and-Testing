#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume spike
# Long when price breaks above Donchian(20) high in uptrend (close > 1d EMA50) with volume spike (>1.8x 20-period avg)
# Short when price breaks below Donchian(20) low in downtrend (close < 1d EMA50) with volume spike
# Exit when price crosses the Donchian middle (10-period average) or trend reverses
# Designed for low trade frequency (~25-45/year) to minimize fee drain. Works in bull/bear by
# combining price channel breakouts with trend filtering and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian middle (average of high and low channels)
    donch_middle = (donch_high + donch_low) / 2.0
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(donch_middle[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        dch_high = donch_high[i]
        dch_low = donch_low[i]
        dch_mid = donch_middle[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if price > dch_high and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif price < dch_low and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Donchian middle or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Donchian middle or trend turns down
                if price < dch_mid or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Donchian middle or trend turns up
                if price > dch_mid or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0