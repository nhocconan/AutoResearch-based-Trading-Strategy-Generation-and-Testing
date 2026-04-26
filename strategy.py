#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA50 trend filter and volume confirmation (>2x average). Uses discrete sizing 0.25 to minimize fee drag. Target: 15-25 trades/year. Works in bull (breakouts with trend) and bear (mean reversion at extremes with volume spike) via tight entry conditions.
"""

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
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) on 1d for breakout confirmation and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    if len(high_1d) < 2:
        camarilla_r1 = np.full_like(close_1d_arr, np.nan)
        camarilla_s1 = np.full_like(close_1d_arr, np.nan)
    else:
        camarilla_r1 = close_1d_arr[:-1] + 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_s1 = close_1d_arr[:-1] - 1.1 * (high_1d[:-1] - low_1d[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels and ATR to 1d timeframe (same as prices for 1d timeframe)
    camarilla_r1_aligned = camarilla_r1  # already 1d aligned
    camarilla_s1_aligned = camarilla_s1  # already 1d aligned
    atr_aligned = atr  # already 1d aligned
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(20, 50, 14)  # volume MA, weekly EMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr_aligned[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_1w_val = ema_50_1w_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        atr_val = atr_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price closes above R1 with weekly uptrend and volume confirmation
            long_signal = (close_val > r1_val) and (close_val > ema_50_1w_val) and volume_confirmed
            # Short: price closes below S1 with weekly downtrend and volume confirmation
            short_signal = (close_val < s1_val) and (close_val < ema_50_1w_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price closes below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: weekly close crosses below EMA50
            elif close_val < ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Stoploss: 2.5*ATR below entry
            elif close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price closes above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Trend reversal: weekly close crosses above EMA50
            elif close_val > ema_50_1w_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 3. Stoploss: 2.5*ATR above entry
            elif close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0