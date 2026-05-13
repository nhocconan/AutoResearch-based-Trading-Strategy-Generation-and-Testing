#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: Hourly Camarilla R1/S1 breakouts with 4h trend (EMA50) and 1d volume confirmation capture intraday momentum.
Works in bull (breakouts with trend) and bear (mean reversion at extremes via trend filter) by using 4h trend filter.
1h timeframe with 4h/1d filters reduces noise; target 15-35 trades/year to avoid fee drag.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
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
    
    # 4h trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d volume filter: current volume > 1.8x 20-day average
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_filter = volume > (1.8 * vol_ma_1d_aligned)
    
    # Hourly Camarilla levels from previous hour
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use previous hour's OHLC to avoid look-ahead
    hourly_high = np.roll(high, 1)
    hourly_low = np.roll(low, 1)
    hourly_close = np.roll(close, 1)
    hourly_high[0] = high[0]
    hourly_low[0] = low[0]
    hourly_close[0] = close[0]
    
    camarilla_r1 = hourly_close + (hourly_high - hourly_low) * 1.1 / 12.0
    camarilla_s1 = hourly_close - (hourly_high - hourly_low) * 1.1 / 12.0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Break above R1 with 4h uptrend and volume confirmation
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Break below S1 with 4h downtrend and volume confirmation
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to Camarilla pivot or 4h trend breaks
            camarilla_pivot = (hourly_high[i] + hourly_low[i] + hourly_close[i]) / 3.0
            if (close[i] < camarilla_pivot) or (close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to Camarilla pivot or 4h trend breaks
            camarilla_pivot = (hourly_high[i] + hourly_low[i] + hourly_close[i]) / 3.0
            if (close[i] > camarilla_pivot) or (close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals