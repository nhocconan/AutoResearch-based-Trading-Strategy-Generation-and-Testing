#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla Pivot Breakout with 1d volume confirmation
    # Uses daily pivot levels (H4/L4) for breakout entries, confirmed by volume spike
    # Works in both bull/bear as it captures institutional breakouts
    # Target: 20-40 trades/year per symbol
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla Pivot levels from previous day
    # H4 = Close + 1.5*(High-Low)
    # L4 = Close - 1.5*(High-Low)
    H4 = close_1d + 1.5 * (high_1d - low_1d)
    L4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align H4/L4 to 4h timeframe (using previous day's values)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_long = close[i] > H4_aligned[i]
        breakout_short = close[i] < L4_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and vol_spike[i]
        short_entry = breakout_short and vol_spike[i]
        
        # Exit conditions: opposite breakout
        long_exit = close[i] < L4_aligned[i]
        short_exit = close[i] > H4_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol"
timeframe = "4h"
leverage = 1.0