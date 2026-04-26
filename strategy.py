#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and volume spike confirmation. Uses wider Camarilla levels (R3/S3) for fewer, higher-quality entries. 1d EMA34 filter ensures trading only in the direction of the daily trend to avoid counter-trend whipsaws. Volume spike confirms institutional participation. Designed for 20-40 trades/year to minimize fee drag while working in both bull and bear markets by aligning with the dominant daily trend.
"""

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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R3 = PP + (high_1d - low_1d) * 1.125 / 4.0  # R3 = PP + (H-L)*1.125/4
    S3 = PP - (high_1d - low_1d) * 1.125 / 4.0  # S3 = PP - (H-L)*1.125/4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(34, 20, 14)  # EMA34, volume avg, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and trend alignment
            # Long: break above R3 + volume spike + price above daily EMA34 (uptrend)
            long_entry = (close_val > R3_aligned[i]) and volume_spike[i] and (close_val > ema_34_1d_aligned[i])
            # Short: break below S3 + volume spike + price below daily EMA34 (downtrend)
            short_entry = (close_val < S3_aligned[i]) and volume_spike[i] and (close_val < ema_34_1d_aligned[i])
            
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
            # Long - exit on S3 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val < S3_aligned[i]) or \
                           (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R3 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val > R3_aligned[i]) or \
                           (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrendFilter_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0