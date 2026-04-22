#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and volume spike
# Long when close > Donchian upper band + close > 12h EMA50 (uptrend) + volume spike
# Short when close < Donchian lower band + close < 12h EMA50 (downtrend) + volume spike
# Exit when price crosses opposite Donchian band or trend reverses
# Designed for low trade frequency (~20-40/year) to minimize fee drain.
# Donchian channels provide clear breakout signals in trending markets, effective in both bull and bear regimes.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper band = highest high over last 20 periods
    # Donchian lower band = lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_val = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above upper band + uptrend + volume spike
            if price > upper and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below lower band + downtrend + volume spike
            elif price < lower and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses opposite band or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price drops below lower band or trend turns down
                if price < lower or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above upper band or trend turns up
                if price > upper or price > ema_val:
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