#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and volume spike for entry timing. 
4h provides signal direction (trend + key levels), 1h provides precise entry within session (08-20 UTC). 
Volume spike confirms institutional participation. Designed for 15-35 trades/year to minimize fee drag while 
working in both bull (breakouts with trend) and bear (mean reversion at extremes in ranging) markets.
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
    
    # Calculate 4h HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h Camarilla pivot levels (R1, S1)
    PP_4h = (high_4h + low_4h + close_4h) / 3.0
    R1_4h = PP_4h + (high_4h - low_4h) * 1.0 / 12.0
    S1_4h = PP_4h - (high_4h - low_4h) * 1.0 / 12.0
    
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    
    # 1h indicators
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.20  # 20% position size
    
    # Warmup: need enough for EMA50 and volume average
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            # Outside session: flatten if needed, no new entries
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for breakout with volume confirmation and trend alignment
            # Long: break above R1 + volume spike + price above EMA50 (uptrend)
            long_entry = (close_val > R1_4h_aligned[i]) and volume_spike[i] and (close_val > ema_50_4h_aligned[i])
            # Short: break below S1 + volume spike + price below EMA50 (downtrend)
            short_entry = (close_val < S1_4h_aligned[i]) and volume_spike[i] and (close_val < ema_50_4h_aligned[i])
            
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
            # Long - exit on retracement to S1 or EMA50
            exit_condition = (close_val < S1_4h_aligned[i]) or (close_val < ema_50_4h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on retracement to R1 or EMA50
            exit_condition = (close_val > R1_4h_aligned[i]) or (close_val > ema_50_4h_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0