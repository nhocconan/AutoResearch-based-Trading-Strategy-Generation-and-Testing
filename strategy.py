#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout confirmed by weekly EMA50 trend and volume spike.
Long when: price breaks above R1 + weekly EMA50 uptrend + volume > 2.0 * 20-day avg volume.
Short when: price breaks below S1 + weekly EMA50 downtrend + volume > 2.0 * 20-day avg volume.
Exit: mean reversion to Camarilla PP (pivot point) or stoploss at 2*ATR.
Uses discrete 0.25 position size. Targets 15-25 trades/year for optimal test generalization.
Works in both bull (breakouts with trend) and bear (mean reversion in ranging markets).
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
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 1d timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for volume avg, 50 for weekly EMA, 14 for ATR
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above R1 + weekly EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below S1 + weekly EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close[i+1] if i+1 < n else close_val  # approximate entry at next open
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close[i+1] if i+1 < n else close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            # Exit 1: mean reversion to PP
            # Exit 2: stoploss at 2*ATR below entry
            exit_pp = close_val < camarilla_pp_aligned[i]
            exit_stop = close_val < entry_price - 2.0 * atr[i]
            
            if exit_pp or exit_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # Exit 1: mean reversion to PP
            # Exit 2: stoploss at 2*ATR above entry
            exit_pp = close_val > camarilla_pp_aligned[i]
            exit_stop = close_val > entry_price + 2.0 * atr[i]
            
            if exit_pp or exit_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0