#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume spike
# Long when price breaks above Donchian upper band + price > 1w EMA50 + volume spike
# Short when price breaks below Donchian lower band + price < 1w EMA50 + volume spike
# Exit when price returns to Donchian midline (mean of upper/lower) or trend reverses
# Designed for low trade frequency (~15-25/year) to minimize fee drain.
# Works in bull/bear by combining breakout momentum with higher timeframe trend filter.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_mid = (dc_high + dc_low) / 2.0  # midline for exit
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: break above upper band + uptrend + volume spike
            if price > dc_high[i] and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + downtrend + volume spike
            elif price < dc_low[i] and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to midline or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midline or trend turns down
                if price < dc_mid[i] or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midline or trend turns up
                if price > dc_mid[i] or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0