#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike
Hypothesis: Use 1h timeframe with Camarilla R1/S1 breakout from prior 4h bar,
confirmed by 4h EMA50 trend and 1d volume spike.
Long when: price breaks above R1 (from prior 4h) + 4h EMA50 uptrend + 1d volume > 2.0 * 20-day avg volume.
Short when: price breaks below S1 (from prior 4h) + 4h EMA50 downtrend + 1d volume > 2.0 * 20-day avg volume.
Exit when: price reverts to prior 4h Camarilla midpoint (PP) or touches opposite level (S1/R1).
Uses discrete 0.20 position size to limit fee drag. Targets 15-37 trades/year on 1h.
Uses 4h for signal direction, 1h only for entry timing precision.
Session filter: 08-20 UTC to reduce noise.
Designed to work in both bull (trend following) and bear (mean reversion via exits) markets.
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
    
    # Calculate Camarilla levels from prior 4h bar (using 4h HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Prior 4h bar's OHLC for Camarilla calculation (shift(1) for completed bar)
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    prev_close = df_4h['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point) from prior 4h bar
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1h timeframe (wait for completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d volume spike: current 1d volume > 2.0 * 20-day avg volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_avg_1d, additional_delay_bars=0)
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_20_avg_1d_aligned)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d, additional_delay_bars=0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for 4h EMA, 20 for 1d volume avg
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.20  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R1 + 4h EMA50 uptrend + 1d volume spike
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and \
                       volume_spike_1d_aligned[i]
            # Short: break below S1 + 4h EMA50 downtrend + 1d volume spike
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and \
                        volume_spike_1d_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S1 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R1 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolSpike"
timeframe = "1h"
leverage = 1.0