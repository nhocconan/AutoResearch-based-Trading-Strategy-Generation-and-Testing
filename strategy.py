#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Keltner_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Keltner channels and trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for middle line
    close_weekly = df_weekly['close'].values
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate weekly ATR for channel width
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_prev = np.roll(close_weekly, 1)
    close_weekly_prev[0] = close_weekly[0]
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - close_weekly_prev)
    tr3 = np.abs(low_weekly - close_weekly_prev)
    tr_weekly = np.maximum(tr1, np.maximum(tr2, tr3))
    atr10_weekly = pd.Series(tr_weekly).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Upper and lower Keltner channels
    keltner_upper = ema20_weekly + 2.0 * atr10_weekly
    keltner_lower = ema20_weekly - 2.0 * atr10_weekly
    
    # Align weekly channels to daily timeframe
    ema20_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_weekly, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_weekly, keltner_lower)
    
    # Daily volume spike filter: current volume > 2.0 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need enough data for weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(keltner_upper_aligned[i]) or 
            np.isnan(keltner_lower_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20 = ema20_aligned[i]
        upper = keltner_upper_aligned[i]
        lower = keltner_lower_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close above upper Keltner + weekly uptrend + volume spike
            if close[i] > upper and close[i] > ema20 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close below lower Keltner + weekly downtrend + volume spike
            elif close[i] < lower and close[i] < ema20 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below EMA20 or upper channel breaks down
            if close[i] < ema20 or close[i] < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above EMA20 or lower channel breaks up
            if close[i] > ema20 or close[i] > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals