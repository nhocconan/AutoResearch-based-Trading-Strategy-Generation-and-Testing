#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume
Hypothesis: 1-hour Camarilla R1/S1 breakouts with 4h trend filter (EMA50), 1d volume confirmation, and session filter (08-20 UTC).
Camarilla levels provide precise support/resistance; trend filter avoids counter-trend trades; volume ensures conviction.
Session filter reduces noise. Targets 15-35 trades/year by using 4h/1d for signal direction and 1h for timing only.
Works in bull/bear via trend filter and volume confirmation.
"""

name = "1h_4h_1d_Camarilla_R1S1_Breakout_Trend_Volume"
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
    
    # Session filter: 08-20 UTC (pre-computed)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Volume confirmation: >1.5x 24-period average (on 1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=24, min_periods=24).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_confirm = volume > (1.5 * vol_ma_1d_aligned)
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R1 = C + (H-L)*1.12/12, S1 = C - (H-L)*1.12/12
    camarilla_multiplier = 1.12 / 12
    camarilla_R1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * camarilla_multiplier
    camarilla_S1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * camarilla_multiplier
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume confirm + price above 4h EMA50
            if (close[i] > camarilla_R1_aligned[i] and 
                volume_confirm[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume confirm + price below 4h EMA50
            elif (close[i] < camarilla_S1_aligned[i] and 
                  volume_confirm[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S1 and R1 OR closes below 4h EMA50
            if (close[i] > camarilla_S1_aligned[i] and close[i] < camarilla_R1_aligned[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters between S1 and R1 OR closes above 4h EMA50
            if (close[i] > camarilla_S1_aligned[i] and close[i] < camarilla_R1_aligned[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals