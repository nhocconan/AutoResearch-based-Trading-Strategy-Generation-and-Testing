#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout + 1w EMA(34) trend filter + volume confirmation.
    # Camarilla levels act as intraday support/resistance; breaks indicate institutional participation.
    # 1w EMA(34) ensures alignment with weekly trend, reducing counter-trend whipsaws.
    # Volume spike (>2.0x 20-period MA) confirms breakout validity.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 50-120 total trades over 4 years (12-30/year) to stay within fee drag limits.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA(34) for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align 1d Camarilla levels and 1w EMA to 1d timeframe (prices is already 1d)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0 * 20-period MA
        volume_filter = volume[i] > 2.0 * volume_ma[i]
        
        # Trend filter: price above/below 1w EMA(34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions (using prior bar's levels)
        long_breakout = (close[i] > camarilla_h4_aligned[i-1]) and volume_filter and uptrend
        short_breakout = (close[i] < camarilla_l4_aligned[i-1]) and volume_filter and downtrend
        
        # Exit conditions: price returns to prior day's close (mean reversion to equilibrium)
        long_exit = close[i] < close_1d[i-1]
        short_exit = close[i] > close_1d[i-1]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_breakout_ema_volume_v1"
timeframe = "1d"
leverage = 1.0