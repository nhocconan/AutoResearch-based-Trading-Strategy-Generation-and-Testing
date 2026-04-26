#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrendFilter_HTFVolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d trend filter (price > EMA34) and 1w volume spike confirmation.
Uses tighter R1/S1 levels for selective entries, 1d EMA34 trend filter to avoid counter-trend trades,
and 1w volume spike to ensure institutional participation. Designed for 20-40 trades/year to minimize fee drag
while working in both bull (trend continuation) and bear (mean reversion at extremes) markets.
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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w volume for spike detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (2.5 * vol_avg_1w)  # 2.5x average = significant spike
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    R1 = PP + (high_1d - low_1d) * 1.0 / 12.0
    S1 = PP - (high_1d - low_1d) * 1.0 / 12.0
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume spike: current volume > 2.0 * 20-period average (4h)
    vol_avg_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_avg_4h)
    
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
    start_idx = max(40, 20, 20, 14)  # EMA34, volume averages, ATR
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike_1w_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size to manage risk
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and trend filter
            # Long: break above R1 + volume spike (4h or 1w) + price above 1d EMA34 (uptrend)
            long_entry = (close_val > R1_aligned[i]) and \
                         (volume_spike_4h[i] or volume_spike_1w_aligned[i]) and \
                         (close_val > ema_34_1d_aligned[i])
            # Short: break below S1 + volume spike (4h or 1w) + price below 1d EMA34 (downtrend)
            short_entry = (close_val < S1_aligned[i]) and \
                          (volume_spike_4h[i] or volume_spike_1w_aligned[i]) and \
                          (close_val < ema_34_1d_aligned[i])
            
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
            # Long - exit on S1 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val < S1_aligned[i]) or \
                           (close_val < entry_price - 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on R1 retracement or ATR stoploss (2.0 * ATR)
            exit_condition = (close_val > R1_aligned[i]) or \
                           (close_val > entry_price + 2.0 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrendFilter_HTFVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0