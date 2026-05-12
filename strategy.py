# NOTE: This is a simplified analysis example using synthetic data.
# In practice, replace the mock data loading with actual data from your exchange or data provider.
# For example, use ccxt to fetch OHLCV data or load from a local database.
# The strategy logic remains the same regardless of the data source.

#!/usr/bin/env python3
"""
1h_4d_Camarilla_R1_S1_Breakout_TrendVol_v1
Hypothesis: 1-hour breakouts from Camarilla R1/S1 levels (based on 4-day price action) with 4-day trend filter and volume spike confirmation.
Targets 1h timeframe for better entry timing while using 4h/1d for signal direction to reduce trade frequency.
Only takes long when price breaks above R1 with volume spike and 4d uptrend, short when breaks below S1 with volume spike and 4d downtrend.
Designed to work in both bull and bear markets via trend filter and volume confirmation to avoid false breakouts.
Focuses on stronger breakout levels (R1/S1) rather than R3/S3 for more frequent but still filtered signals.
Uses session filter (08-20 UTC) to avoid low-volume, noisy periods.
Target: 15-37 trades/year per symbol.
"""

name = "1h_4d_Camarilla_R1_S1_Breakout_TrendVol_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 50-period average (on 1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Avoid look-ahead: only use previous 4h bar's data
    range_ = prev_high - prev_low
    R1 = prev_close + 1.1 * range_ / 6
    S1 = prev_close - 1.1 * range_ / 6
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d EMA50 for stronger trend filter (optional, for additional confirmation)
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 4h EMA34 + price above 1d EMA50
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_4h_aligned[i] and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 4h EMA34 + price below 1d EMA50
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_4h_aligned[i] and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S1 and R1 OR closes below 4h EMA34 OR below 1d EMA50
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] < ema_34_4h_aligned[i] or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters between S1 and R1 OR closes above 4h EMA34 OR above 1d EMA50
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] > ema_34_4h_aligned[i] or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals