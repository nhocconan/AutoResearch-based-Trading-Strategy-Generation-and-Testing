#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d trend filter and volume confirmation.
12h timeframe reduces trade frequency to avoid fee drag. Camarilla levels provide precise support/resistance.
1d EMA34 trend filter ensures breakouts align with daily trend. Volume spike confirms institutional participation.
ATR-based stoploss manages risk. Designed for both bull and bear markets by combining breakout with trend filter.
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
    
    # Calculate 12h Camarilla pivot levels (R3, R2, R1, PP, S1, S2, S3)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla pivot calculations
    PP = (high_12h + low_12h + close_12h) / 3.0
    R1 = PP + (high_12h - low_12h) * 1.0 / 12.0
    R2 = PP + (high_12h - low_12h) * 2.0 / 12.0
    R3 = PP + (high_12h - low_12h) * 3.0 / 12.0
    S1 = PP - (high_12h - low_12h) * 1.0 / 12.0
    S2 = PP - (high_12h - low_12h) * 2.0 / 12.0
    S3 = PP - (high_12h - low_12h) * 3.0 / 12.0
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed for pivot levels)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    R2_aligned = align_htf_to_ltf(prices, df_12h, R2)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    S2_aligned = align_htf_to_ltf(prices, df_12h, S2)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
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
    
    # Warmup: need enough for all indicators
    start_idx = max(20, 20, 14, 34)  # volume avg, ATR, EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R3 + price above 1d EMA34 + volume spike
            long_entry = (close_val > R3_aligned[i]) and \
                       (close_val > ema_34_1d_aligned[i]) and \
                       volume_spike[i]
            # Short: break below S3 + price below 1d EMA34 + volume spike
            short_entry = (close_val < S3_aligned[i]) and \
                        (close_val < ema_34_1d_aligned[i]) and \
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
            # Long - exit on R1 retracement or ATR stoploss
            exit_condition = (close_val < R1_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on S1 retracement or ATR stoploss
            exit_condition = (close_val > S1_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0