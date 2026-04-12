#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_keltner_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner Channel (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 and ATR(10) for Keltner Channel
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr10_1w = pd.Series(high_1w - low_1w).rolling(window=10, min_periods=10).mean().values
    
    upper_keltner = ema20_1w + (2.0 * atr10_1w)
    lower_keltner = ema20_1w - (2.0 * atr10_1w)
    
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1w, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1w, lower_keltner)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long breakout above upper Keltner with weekly uptrend and volume
        long_signal = (close[i] > upper_keltner_aligned[i] and 
                      ema20_1w_aligned[i] > ema20_1w_aligned[max(0, i-1)] and
                      vol_confirm[i])
        
        # Short breakdown below lower Keltner with weekly downtrend and volume
        short_signal = (close[i] < lower_keltner_aligned[i] and 
                       ema20_1w_aligned[i] < ema20_1w_aligned[max(0, i-1)] and
                       vol_confirm[i])
        
        # Exit when price crosses back through the 20 EMA
        exit_long = close[i] < ema20_1w_aligned[i]
        exit_short = close[i] > ema20_1w_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals