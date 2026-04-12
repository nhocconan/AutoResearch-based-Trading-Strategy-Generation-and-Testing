#/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Trend
Hypothesis: Camarilla breakout on 12h with 1w trend filter and 1d volume confirmation.
Only long in weekly uptrend, short in weekly downtrend.
Volume must confirm breakout on daily timeframe.
Designed to work in both bull and bear by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = align_htf_to_ltf(prices, df_1w, ema_50_1w) < close_1w[-1] if len(close_1w) > 0 else True
    # Actually compute properly:
    ema_50_1w_series = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean()
    weekly_uptrend_arr = ema_50_1w_series.values
    weekly_uptrend = align_htf_to_ltf(prices, df_1w, weekly_uptrend_arr)
    weekly_downtrend = weekly_uptrend_arr > close_1w[-1] if len(close_1w) > 0 else False
    weekly_downtrend = align_htf_to_ltf(prices, df_1w, 
        pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values) > close
    # Simpler: weekly trend = price > EMA50
    weekly_trend = align_htf_to_ltf(prices, df_1w, 
        pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values)
    weekly_uptrend = close > weekly_trend
    weekly_downtrend = close < weekly_trend
    
    # === DAILY DATA FOR VOLUME CONFIRMATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 12H DATA FOR CAMARILLA BREAKOUT ===
    # Camarilla levels based on previous day
    # Need to calculate from daily OHLC but align to 12h
    # Simpler: use intraday Camarilla on 12h itself
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # But we need previous period's range
    
    # Calculate Camarilla from previous 12h bar
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    # True range equivalent for Camarilla
    rng = high_shift - low_shift
    camarilla_high = close_shift + 1.1 * rng / 2
    camarilla_low = close_shift - 1.1 * rng / 2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(weekly_uptrend[i]) if isinstance(weekly_uptrend, np.ndarray) else False) or \
           np.isnan(vol_ratio_1d_aligned[i]) or \
           np.isnan(camarilla_high[i]) or np.isnan(camarilla_low[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        volume_confirm = vol_ratio_1d_aligned[i] > 1.5
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_high[i]
        breakout_down = close[i] < camarilla_low[i]
        
        # Trend filter: only trade in direction of weekly trend
        long_setup = breakout_up and volume_confirm and weekly_uptrend[i]
        short_setup = breakout_down and volume_confirm and weekly_downtrend[i]
        
        # Exit on opposite Camarilla touch or volume failure
        exit_long = close[i] < camarilla_low[i] or vol_ratio_1d_aligned[i] < 1.0
        exit_short = close[i] > camarilla_high[i] or vol_ratio_1d_aligned[i] < 1.0
        
        # Execute trades
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals