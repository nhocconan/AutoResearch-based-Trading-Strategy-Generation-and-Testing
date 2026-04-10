#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ATR(14) trailing stop
# - Donchian breakout on 12h: long when price > highest high of last 20 bars, short when price < lowest low
# - Volume filter: current 12h volume > 1.5x 20-period average volume on 1d timeframe (avoid low-volume breakouts)
# - ATR trailing stop: exit long when price drops 2.5*ATR from highest high since entry, exit short when price rises 2.5*ATR from lowest low
# - Discrete position sizing: 0.25 long/short to limit drawdown and reduce fee churn
# - Target: 20-40 trades/year (80-160 total over 4 years) to stay within HARD MAX: 200 total
# - Works in bull markets via breakout continuation, works in bear markets via breakdown continuation
# - Volume confirmation filters false breakouts, ATR stop manages risk

name = "12h_1d_donchian_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Rolling highest high and lowest low for Donchian channels
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 1d volume and its 20-period moving average for confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 12h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 12h ATR(14) for trailing stop
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average volume from 1d
        volume_12h_current = prices['volume'].iloc[i]
        volume_spike = volume_12h_current > 1.5 * volume_ma_aligned[i]
        
        close_price = close_12h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price > highest high of last 20 bars with volume confirmation
            if close_price > highest_high[i] and volume_spike:
                position = 1
                highest_since_entry = high_12h[i]  # initialize with current high
                signals[i] = 0.25
            # Short breakdown: price < lowest low of last 20 bars with volume confirmation
            elif close_price < lowest_low[i] and volume_spike:
                position = -1
                lowest_since_entry = low_12h[i]   # initialize with current low
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit via ATR trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, high_12h[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = close_price < highest_since_entry - 2.5 * atr_12h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, low_12h[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = close_price > lowest_since_entry + 2.5 * atr_12h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals