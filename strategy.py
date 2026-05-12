#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # 4h data for Camarilla levels (H4, L4)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's Camarilla R1, S1 (requires previous bar close)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We need previous 4h bar's H, L, C
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    prev_close_4h[0] = np.nan
    
    camarilla_r1_4h = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    camarilla_s1_4h = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # 1d data for trend filter (EMA34) and volume spike
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA(34) on daily close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Volume average (20-period)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 1h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_4h_aligned[i]) or np.isnan(camarilla_s1_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + price > EMA34 + volume > 1.5x avg
            if (close[i] > camarilla_r1_4h_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + price < EMA34 + volume > 1.5x avg
            elif (close[i] < camarilla_s1_4h_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below S1 or price < EMA34
            if close[i] < camarilla_s1_4h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above R1 or price > EMA34
            if close[i] > camarilla_r1_4h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals