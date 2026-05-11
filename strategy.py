#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Current period data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter: current volume > 1.5 * 20-period SMA volume
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_sma * 1.5)
    
    # Calculate Camarilla levels from previous day
    # Previous day's OHLC
    prev_day_high = np.roll(high, 1)
    prev_day_low = np.roll(low, 1)
    prev_day_close = np.roll(close, 1)
    prev_day_high[0] = np.nan
    prev_day_low[0] = np.nan
    prev_day_close[0] = np.nan
    
    # Camarilla R1 and S1
    camarilla_r1 = prev_day_close + (1.1/12) * (prev_day_high - prev_day_low)
    camarilla_s1 = prev_day_close - (1.1/12) * (prev_day_high - prev_day_low)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume spike and above 12h EMA50
            if close[i] > camarilla_r1[i] and vol_spike[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and below 12h EMA50
            elif close[i] < camarilla_s1[i] and vol_spike[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below S1
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above R1
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals