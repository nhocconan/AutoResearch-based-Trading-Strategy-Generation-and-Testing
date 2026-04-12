#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_dema_cross_v1
# Uses 4h DEMA crossover for trend direction (21/55) and 1h for entry timing.
# Enters long when 4h DEMA21 > DEMA55 and price crosses above 1h VWAP.
# Enters short when 4h DEMA21 < DEMA55 and price crosses below 1h VWAP.
# Includes session filter (08-20 UTC) to reduce noise.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drag.
# Works in bull markets (trend following) and bear markets (trend following with filters).

name = "1h_4h_dema_cross_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for DEMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:
        return np.zeros(n)
    
    # Calculate DEMA on 4h close
    close_4h = df_4h['close'].values
    ema1_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema2_4h = pd.Series(close_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    # DEMA = 2*EMA - EMA(EMA)
    ema_of_ema1_4h = pd.Series(ema1_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_of_ema2_4h = pd.Series(ema2_4h).ewm(span=55, adjust=False, min_periods=55).mean().values
    dema21_4h = 2 * ema1_4h - ema_of_ema1_4h
    dema55_4h = 2 * ema2_4h - ema_of_ema2_4h
    
    # Align 4h DEMA to 1h timeframe (wait for 4h bar close)
    dema21_4h_aligned = align_htf_to_ltf(prices, df_4h, dema21_4h)
    dema55_4h_aligned = align_htf_to_ltf(prices, df_4h, dema55_4h)
    
    # 1h VWAP for entry timing
    typical_price = (high + low + close) / 3.0
    vwap_num = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    vwap_den = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, 0.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(55, n):
        # Skip if not in session or data not ready
        if not session_filter[i] or np.isnan(dema21_4h_aligned[i]) or np.isnan(dema55_4h_aligned[i]) or np.isnan(vwap[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h DEMA21 > DEMA55 for long, < for short
        trend_long = dema21_4h_aligned[i] > dema55_4h_aligned[i]
        trend_short = dema21_4h_aligned[i] < dema55_4h_aligned[i]
        
        # Entry signals: price crosses VWAP in direction of 4h trend
        if trend_long and close[i] > vwap[i] and close[i-1] <= vwap[i-1] and position != 1:
            position = 1
            signals[i] = 0.20
        elif trend_short and close[i] < vwap[i] and close[i-1] >= vwap[i-1] and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit signals: price crosses VWAP against trend or trend reversal
        elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1]) or (not trend_long and position == 1):
            position = 0
            signals[i] = 0.0
        elif (close[i] > vwap[i] and close[i-1] <= vwap[i-1]) or (not trend_short and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals