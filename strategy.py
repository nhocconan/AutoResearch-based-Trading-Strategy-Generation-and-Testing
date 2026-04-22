#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets. 1d EMA34 filters direction: 
# only long when price > 1d EMA34 (uptrend), short when price < 1d EMA34 (downtrend).
# Volume confirmation (current volume > 2x 20-period average) avoids false breakouts.
# Designed for 6h timeframe to balance trade frequency (~20-50/year) and capture major moves.
# Works in bull/bear markets by aligning with higher timeframe trend via 1d EMA34.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d data
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 6h data (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_1d_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict to reduce false signals)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above upper band + uptrend + volume spike
            if price > highest_high[i] and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below lower band + downtrend + volume spike
            elif price < lowest_low[i] and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: return to midpoint or trend reversal
            midpoint = (highest_high[i] + lowest_low[i]) / 2.0
            
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to midpoint or trend breaks down
                if price < midpoint or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to midpoint or trend breaks up
                if price > midpoint or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0