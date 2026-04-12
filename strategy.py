#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h strategy using 4h Camarilla pivot breakout with volume confirmation and 1d trend filter
    # Uses 4h Camarilla levels (H3, L3) for breakout entries in direction of 1d EMA(50) trend
    # Volume > 1.3x 20-period MA confirms breakout strength
    # Session filter: 08-20 UTC to avoid low-volume Asian session noise
    # Discrete position sizing: 0.20 long/short to minimize fee churn
    # Target: 15-37 trades/year (60-150 over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels (based on previous 4h bar)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_h3 = close_4h + 1.1 * (high_4h - low_4h)
    camarilla_l3 = close_4h - 1.1 * (high_4h - low_4h)
    
    # Align Camarilla levels to 1h timeframe (delayed by one 4h bar for completed bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.3x 20-period MA
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    
    vol_ratio = np.full(n, np.nan)
    for i in range(20, n):
        if vol_ma_20[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma_20[i]
        else:
            vol_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA(50)
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > camarilla_h3_aligned[i]
        breakout_short = close[i] < camarilla_l3_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_long and (vol_ratio[i] > 1.3) and uptrend
        short_entry = breakout_short and (vol_ratio[i] > 1.3) and downtrend
        
        # Exit conditions: opposite breakout or volume drought
        long_exit = breakout_short or (vol_ratio[i] < 0.7)
        short_exit = breakout_long or (vol_ratio[i] < 0.7)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_camarilla_breakout_vol_session_v1"
timeframe = "1h"
leverage = 1.0