#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Use 6h timeframe with Camarilla R3/S3 breakout confirmed by 1d EMA34 trend and volume spike. Camarilla levels from 1d provide institutional support/resistance. Breakouts above R3 or below S3 with trend and volume confirmation capture strong moves. Targets 12-37 trades/year to minimize fee drag. Works in bull/bear markets by using 1d EMA34 for trend direction and volume confirmation to filter false breakouts.
"""

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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's high, low, close
    df_1d_prev = df_1d.copy()
    df_1d_prev['high'] = df_1d_prev['high'].shift(1)
    df_1d_prev['low'] = df_1d_prev['low'].shift(1)
    df_1d_prev['close'] = df_1d_prev['close'].shift(1)
    
    # Camarilla R3, S3, R4, S4
    # R4 = Close + 1.5*(High-Low)
    # R3 = Close + 1.125*(High-Low)
    # S3 = Close - 1.125*(High-Low)
    # S4 = Close - 1.5*(High-Low)
    h = df_1d_prev['high'].values
    l = df_1d_prev['low'].values
    c = df_1d_prev['close'].values
    
    camarilla_r3 = c + 1.125 * (h - l)
    camarilla_s3 = c - 1.125 * (h - l)
    camarilla_r4 = c + 1.5 * (h - l)
    camarilla_s4 = c - 1.5 * (h - l)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_s4)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1d EMA, 20 for volume avg, 14 for ATR
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Camarilla R3 + 1d EMA34 uptrend + volume spike
            long_entry = (close_val > camarilla_r3_aligned[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Camarilla S3 + 1d EMA34 downtrend + volume spike
            short_entry = (close_val < camarilla_s3_aligned[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Camarilla S3 break or ATR stoploss
            exit_condition = (close_val < camarilla_s3_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla R3 break or ATR stoploss
            exit_condition = (close_val > camarilla_r3_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0