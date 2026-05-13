#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Trend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) breakouts with 4h trend (EMA50) and volume confirmation work in both bull and bear markets.
Breakout above R1 with 4h uptrend and volume spike = long.
Breakdown below S1 with 4h downtrend and volume spike = short.
Exit on opposite level touch or trend reversal. Uses 1d trend filter for higher timeframe bias.
Target: 20-50 trades/year per symbol. Uses session filter (08-20 UTC) to reduce noise.
"""

name = "1h_Camarilla_R1_S1_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla pivot calculation (based on previous day)
    # For each hour, we use the previous day's high, low, close
    # Since we're on 1h timeframe, we need to calculate daily pivots
    # We'll calculate pivots once per day and use them for all hours of that day
    
    # Convert to daily data for pivot calculation
    # We'll use the same method as in the data: group by day
    df = prices.copy()
    df['date'] = df['open_time'].dt.date
    
    # Calculate daily high, low, close
    daily = df.groupby('date').agg({
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).reset_index()
    
    if len(daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    daily['range'] = daily['high'] - daily['low']
    daily['R1'] = daily['close'] + daily['range'] * 1.1 / 12
    daily['S1'] = daily['close'] - daily['range'] * 1.1 / 12
    
    # Map daily levels back to hourly data
    # Create mapping from date to R1, S1
    r1_map = dict(zip(daily['date'], daily['R1']))
    s1_map = dict(zip(daily['date'], daily['S1']))
    
    # Apply to each hour
    r1 = np.array([r1_map.get(date, np.nan) for date in df['date']])
    s1 = np.array([s1_map.get(date, np.nan) for date in df['date']])
    
    # Forward fill to handle any missing values (shouldn't happen with proper data)
    # But we'll handle NaN by using previous valid value
    for i in range(1, len(r1)):
        if np.isnan(r1[i]):
            r1[i] = r1[i-1]
        if np.isnan(s1[i]):
            s1[i] = s1[i-1]
    
    # 4h trend: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Skip if pivot levels not available
        if np.isnan(r1[i]) or np.isnan(s1[i]):
            signals[i] = 0.0
            continue
            
        # Get values
        r1_level = r1[i]
        s1_level = s1[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R1, 4h uptrend, 1d uptrend filter, volume confirmation, in session
            if close[i] > r1_level and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.20
                position = 1
            # SHORT: break below S1, 4h downtrend, 1d downtrend filter, volume confirmation, in session
            elif close[i] < s1_level and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: touch S1 or 4h trend turns down
            if close[i] < s1_level or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: touch R1 or 4h trend turns up
            if close[i] > r1_level or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals