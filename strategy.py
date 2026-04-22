#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA trend filter and volume confirmation
# Long when close breaks above Donchian upper band (20) + close > 1w EMA50 + volume spike
# Short when close breaks below Donchian lower band (20) + close < 1w EMA50 + volume spike
# Exit when price crosses 10-period SMA or trend reverses
# Donchian channels capture breakouts in trending markets; EMA filter ensures direction alignment
# Volume spike confirms breakout strength. Designed for low trade frequency (~10-25/year on 1d).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on 1w close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 10-period SMA for exit
    close = prices['close'].values
    sma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Donchian channels (20-period) on 1d data
    high = prices['high'].values
    low = prices['low'].values
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(sma_10[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema_val = ema_50_aligned[i]
        sma_val = sma_10[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
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
            # Exit conditions: price crosses 10-period SMA or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price drops below SMA or trend turns down
                if price < sma_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price rises above SMA or trend turns up
                if price > sma_val or price > ema_val:
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