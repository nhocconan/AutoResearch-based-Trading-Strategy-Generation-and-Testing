#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Donchian breakout captures trend continuation; 1w EMA ensures alignment with higher-timeframe trend.
# Volume filter requires current volume > 1.3x 20-day average to avoid false breakouts.
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in both bull/bear markets.
# Only takes longs when price > 1w EMA, shorts when price < 1w EMA.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on 1w data
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channels (20-period) on daily data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or 
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
        ema_val = ema_1w_aligned[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_confirm = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long conditions: breakout above upper Donchian + uptrend + volume confirmation
            if price > highest_high[i] and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below lower Donchian + downtrend + volume confirmation
            elif price < lowest_low[i] and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: return to opposite Donchian band or trend reversal
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to or below lower Donchian or trend breaks
                if price <= lowest_low[i] or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to or above upper Donchian or trend breaks
                if price >= highest_high[i] or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0