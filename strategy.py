#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA34 for 1d trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels using previous day's OHLC
    # For intraday calculation, we need to group by date
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_prev = daily['high'].shift(1).values
    low_prev = daily['low'].shift(1).values
    close_prev = daily['close'].shift(1).values
    
    # Camarilla R1, S1 levels
    R1 = close_prev + (high_prev - low_prev) * 1.0 / 12
    S1 = close_prev - (high_prev - low_prev) * 1.0 / 12
    
    # Map daily levels to intraday bars
    date_to_idx = {date: i for i, date in enumerate(daily['date'])}
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    
    for i, row in df.iterrows():
        date = row['date']
        if date in date_to_idx and date_to_idx[date] > 0:  # Ensure we have previous day
            idx = date_to_idx[date]
            camarilla_R1[i] = R1[idx-1]
            camarilla_S1[i] = S1[idx-1]
    
    # Volume spike detection (24-period average for 4h)
    vol_ma_4h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma_4h * 2.0)
    
    # Align all indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_R1_aligned = align_htf_to_ltf(prices, daily, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, daily, camarilla_S1)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 24)  # Ensure enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R1, price above 1d EMA34, and volume spike
            if (close[i] > camarilla_R1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1, price below 1d EMA34, and volume spike
            elif (close[i] < camarilla_S1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla S1 or price below 1d EMA34
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla R1 or price above 1d EMA34
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals