#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for HTF context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h high/low/close for calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h range for pivot calculations
    range_12h = high_12h - low_12h
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range for pivot calculations
    range_1d = high_1d - low_1d
    
    # 12h Camarilla pivot levels (based on previous 12h bar)
    camarilla_r4_12h = close_12h + range_12h * 1.1 / 2
    camarilla_s4_12h = close_12h - range_12h * 1.1 / 2
    
    # Daily Camarilla pivot levels (based on previous day)
    camarilla_r4_1d = close_1d + range_1d * 1.1 / 2
    camarilla_s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Daily EMA34 for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h Camarilla levels and daily EMA to 4h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Daily trend filter: price above/below daily EMA34
        price_above_daily_ema = close[i] > ema_34_1d_aligned[i]
        price_below_daily_ema = close[i] < ema_34_1d_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above 12h R4 with volume, daily uptrend
        # Short: price breaks below 12h S4 with volume, daily downtrend
        long_entry = (close[i] > r4_12h_aligned[i]) and price_above_daily_ema and vol_filter
        short_entry = (close[i] < s4_12h_aligned[i]) and price_below_daily_ema and vol_filter
        
        # Exit conditions: price returns to opposite daily S4/R4 levels
        long_exit = (close[i] < s4_1d_aligned[i])
        short_exit = (close[i] > r4_1d_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4S4_12hPivot_DailyEMA34"
timeframe = "4h"
leverage = 1.0