#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend filter + volume spike
# Uses Donchian channel breakouts for momentum capture with weekly EMA trend filter
# Volume spike confirmation reduces false breakouts
# ATR-based trailing stop (3.0) lets winners run while cutting losses
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets by buying breakouts above upper channel in uptrend
# Works in bear markets by selling breakdowns below lower channel in downtrend
# Weekly EMA50 filter ensures we only trade with the dominant trend
# Volume spike (2.0x 20-period average) confirms breakout strength

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # For long trailing stop
    lowest_low_since_entry = 0.0    # For short trailing stop
    
    # Start after warmup (need enough for Donchian, ATR, EMA, and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Uptrend: price > weekly EMA50
            # Downtrend: price < weekly EMA50
            uptrend = close[i] > ema_50_1w_aligned[i]
            downtrend = close[i] < ema_50_1w_aligned[i]
            
            # Long: Price breaks above upper Donchian + volume spike + uptrend
            if close[i] > highest_high[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: Price breaks below lower Donchian + volume spike + downtrend
            elif close[i] < lowest_low[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if high[i] > highest_high_since_entry:
                highest_high_since_entry = high[i]
            
            # ATR trailing stop: exit if price drops 3.0*ATR from highest high
            trailing_stop = highest_high_since_entry - 3.0 * atr[i]
            if close[i] < trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if low[i] < lowest_low_since_entry:
                lowest_low_since_entry = low[i]
            
            # ATR trailing stop: exit if price rises 3.0*ATR from lowest low
            trailing_stop = lowest_low_since_entry + 3.0 * atr[i]
            if close[i] > trailing_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals