#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Camarilla R1/S1 breakout on 12h with 1w EMA50 trend filter and volume spike (>1.5x average volume). Uses ATR-based stoploss (2.0*ATR) and discrete position sizing (0.25) to minimize fee churn. Designed for lower trade frequency on 12h timeframe to avoid overtrading while capturing strong breakouts in both bull and bear markets by following the 1w trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for EMA, volume, ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for Camarilla levels (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (inner levels, more reliable breakouts)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (1 bar delay for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate average volume for confirmation (30-period SMA for 12h)
    avg_volume = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0
    
    # Start after warmup (need 50 for EMA, 30 for volume, 14 for ATR)
    start_idx = max(50, 30, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (strong breakout)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Camarilla R1 with 1w uptrend and volume confirmation
        long_condition = (close_val > r1_val) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Camarilla S1 with 1w downtrend and volume confirmation
        short_condition = (close_val < s1_val) and (close_val < ema_val) and volume_confirmed
        
        # Stoploss logic: price moves against position by atr_multiplier * ATR from entry
        long_stop = (position == 1 and close_val < entry_price - atr_multiplier * atr_val)
        short_stop = (position == -1 and close_val > entry_price + atr_multiplier * atr_val)
        
        # Exit logic: 
        # Long exit: price retests or breaks below Camarilla R1 (failed breakout) OR stoploss hit
        long_exit = (position == 1 and (close_val <= r1_val or long_stop))
        # Short exit: price retests or breaks above Camarilla S1 (failed breakout) OR stoploss hit
        short_exit = (position == -1 and (close_val >= s1_val or short_stop))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0