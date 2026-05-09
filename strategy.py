#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h Camarilla R1/S1 breakout with 1d EMA trend filter and volume confirmation.
    Long: Close breaks above R1 and above 1d EMA(34) and volume > 1.5x 20-period avg
    Short: Close breaks below S1 and below 1d EMA(34) and volume > 1.5x 20-period avg
    Exit: Opposite signal or price crosses EMA(34)
    Position size: 0.25
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # For each 4h bar, use previous day's OHLC to calculate today's Camarilla levels
    # We'll calculate daily OHLC first, then align to 4h
    
    # Get daily OHLC from 1d data
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    rang = daily_high - daily_low
    r1_daily = daily_close + rang * 1.1 / 12
    s1_daily = daily_close - rang * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe (wait for day to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_daily)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_daily)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(daily_close)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above 1d EMA trend, and volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1, below 1d EMA trend, and volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA(34) or breaks below S1
            if close[i] < ema34_1d_aligned[i] or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA(34) or breaks above R1
            if close[i] > ema34_1d_aligned[i] or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals