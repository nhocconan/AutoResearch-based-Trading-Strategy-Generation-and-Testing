#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data for longer trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA20 for trend filter
    ema20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Calculate 1w EMA20 for longer trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 6h Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low)
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema20_1d_val = ema20_1d_aligned[i]
        ema20_1w_val = ema20_1w_aligned[i]
        price = close[i]
        
        # Trend filter: both 1d and 1w EMA20 must agree
        uptrend = price > ema20_1d_val and price > ema20_1w_val
        downtrend = price < ema20_1d_val and price < ema20_1w_val
        
        if position == 0:
            # Long: price breaks above 6h Donchian high + both timeframes uptrend
            if price > donch_high_val and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian low + both timeframes downtrend
            elif price < donch_low_val and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low
                if price < donch_low_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high
                if price > donch_high_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1d1wEMA20_Trend"
timeframe = "6h"
leverage = 1.0