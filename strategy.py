#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and session filter (08-20 UTC).
# Long when price breaks above R1 with price > 1d EMA50 (bullish trend) and within active session.
# Short when price breaks below S1 with price < 1d EMA50 (bearish trend) and within active session.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts).
# Uses 4h for signal direction (structure) and 1d EMA50 for regime filter to reduce false breakouts.
# Session filter reduces noise during low-activity periods. Position size 0.20 to limit drawdown.
# Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits for 1h timeframe.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dEMA50_Session_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # --- 4h Indicators (MTF for direction) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Precompute prior day's OHLC for Camarilla (prior 4h day = prior 1d)
    open_time_4h = df_4h['open_time']
    prior_day_start = open_time_4h - pd.Timedelta(days=1)
    prior_day_start = prior_day_start.dt.normalize()
    
    # Get 1d data for prior day OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    df_1d = df_1d.copy()
    df_1d['date'] = df_1d['open_time'].dt.date
    prior_day_start_date = prior_day_start.dt.date
    
    # Create mapping from date to OHLC
    ohlc_map = df_1d.groupby('date').agg({
        'high': 'first',
        'low': 'first',
        'close': 'first'
    })
    
    camarilla_r1_4h = np.full(len(df_4h), np.nan)
    camarilla_s1_4h = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        pd_date = prior_day_start_date.iloc[i]
        if pd_date in ohlc_map.index:
            day_data = ohlc_map.loc[pd_date]
            high_val = day_data['high']
            low_val = day_data['low']
            close_val = day_data['close']
            range_val = high_val - low_val
            camarilla_r1_4h[i] = close_val + (range_val * 1.1 / 12)  # R1
            camarilla_s1_4h[i] = close_val - (range_val * 1.1 / 12)  # S1
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # --- 1d Indicators (MTF for trend filter) ---
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # --- Session filter (08-20 UTC) ---
    # open_time is already datetime64[ms], use index for hour
    hours = prices.index.hour  # Pre-compute before loop
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data or outside session
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(camarilla_r1_4h_aligned[i]) or
            np.isnan(camarilla_s1_4h_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + price > 1d EMA50 (bullish trend)
            if (close[i] > camarilla_r1_4h_aligned[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + price < 1d EMA50 (bearish trend)
            elif (close[i] < camarilla_s1_4h_aligned[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1
            if close[i] < camarilla_s1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1
            if close[i] > camarilla_r1_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals