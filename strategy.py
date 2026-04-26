#!/usr/bin/env python3
"""
4h_Camarilla_R2_S2_Breakout_1dEMA50_Trend_VolumeSpike_v1
Hypothesis: Camarilla R2/S2 breakout on 4h with 1d EMA50 trend filter and volume spike confirmation.
Uses wider Camarilla levels (R2/S2) to capture stronger moves while reducing noise and overtrading.
1d EMA50 determines trend direction: price above EMA50 = bullish bias (long only), price below EMA50 = bearish bias (short only).
Volume spike confirms institutional participation. Designed for 15-35 trades/year to minimize fee drag
while working in both bull and bear markets by taking directional trades only.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla pivot levels (R2, S2)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R2 = PP + (high_1d - low_1d) * 2.0 / 12.0
    S2 = PP - (high_1d - low_1d) * 2.0 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    
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
    start_idx = max(50, 20, 14)  # EMA, volume avg, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R2_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        ema_trend = ema_1d_aligned[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout in direction of 1d trend with volume confirmation
            # Long: price above 1d EMA50 AND break above R2 + volume spike
            long_entry = (close_val > ema_trend) and (close_val > R2_aligned[i]) and volume_spike[i]
            # Short: price below 1d EMA50 AND break below S2 + volume spike
            short_entry = (close_val < ema_trend) and (close_val < S2_aligned[i]) and volume_spike[i]
            
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
            # Long - exit on S2 retracement or ATR stoploss
            exit_condition = (close_val < S2_aligned[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R2 retracement or ATR stoploss
            exit_condition = (close_val > R2_aligned[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0