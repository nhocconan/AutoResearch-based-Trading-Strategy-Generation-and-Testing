#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_V1
Strategy: 4h Camarilla pivot R1/S1 breakout with volume confirmation and daily EMA34 trend filter.
Long: Price breaks above R1 + volume > 1.5x 20-period avg + price > daily EMA34
Short: Price breaks below S1 + volume > 1.5x 20-period avg + price < daily EMA34
Exit: Opposite pivot level touch or trend reversal
Position size: 0.25
Designed to capture breakouts in trending markets while avoiding false signals in ranging conditions.
Timeframe: 4h
"""

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
    
    # Calculate Camarilla pivot levels for current day using previous day's OHLC
    def calculate_camarilla(high, low, close):
        # Typical price
        pp = (high + low + close) / 3
        # Range
        range_ = high - low
        # Camarilla levels
        r1 = pp + range_ * 1.1 / 12
        s1 = pp - range_ * 1.1 / 12
        return r1, s1
    
    # Need previous day's data to calculate today's levels
    # We'll calculate for each bar using previous day's OHLC
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    # Convert to pandas for easier date handling
    df = prices.copy()
    df['date'] = pd.to_datetime(df['open_time']).dt.date
    
    # Group by date to get daily OHLC
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day (using previous day's data)
    daily['r1'] = np.nan
    daily['s1'] = np.nan
    
    for i in range(1, len(daily)):
        prev_high = daily.iloc[i-1]['high']
        prev_low = daily.iloc[i-1]['low']
        prev_close = daily.iloc[i-1]['close']
        r1_val, s1_val = calculate_camarilla(prev_high, prev_low, prev_close)
        daily.iloc[i, daily.columns.get_loc('r1')] = r1_val
        daily.iloc[i, daily.columns.get_loc('s1')] = s1_val
    
    # Map daily levels back to 4h bars
    date_map = dict(zip(daily['date'], zip(daily['r1'], daily['s1'])))
    for i in range(n):
        bar_date = pd.to_datetime(df.iloc[i]['open_time']).date()
        if bar_date in date_map:
            r1[i], s1[i] = date_map[bar_date]
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.5 * volume_ma20_4h_aligned[i])
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1[i]
        breakout_down = close[i] < s1[i]
        
        # Entry signals
        if position == 0:
            # Long: breakout above R1 + volume filter + trend up
            if breakout_up and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + volume filter + trend down
            elif breakout_down and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches S1 or trend down
            if close[i] <= s1[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches R1 or trend up
            if close[i] >= r1[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_V1"
timeframe = "4h"
leverage = 1.0