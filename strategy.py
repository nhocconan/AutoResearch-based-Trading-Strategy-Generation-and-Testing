#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_v1
Hypothesis: Daily Camarilla R1/S1 breakout with weekly EMA34 trend filter and volume spike captures institutional moves. Uses discrete sizing (0.30) and ATR stoploss (2.0) with 3-day minimum hold. Only takes breakouts in weekly trend direction to work in both bull and bear markets. Targets 20-50 trades over 4 years (5-12/year) to minimize fee drag.
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
    
    # Get 1w data for trend filter (EMA34) - HTF
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need for EMA34
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1w ATR for stoploss calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]))
    tr2 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Get 1d data for Camarilla calculation (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Camarilla levels based on previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r1 = prev_close + 0.5 * (prev_high - prev_low)
    camarilla_s1 = prev_close - 0.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1d timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: volume > 2.5x 20-period average (balanced for frequency/quality)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    days_since_entry = 0
    
    # Start index: need warmup for calculations
    start_idx = max(34, 20, 14)  # EMA34 needs 34, vol MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Get aligned values
        ema_34_val = ema_34_1w_aligned[i]
        atr_val = atr_1w_aligned[i]
        pp_val = camarilla_pp_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Get 1d close aligned for direct comparison
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_val = close_1d_aligned[i]
        is_uptrend = close_1d_val > ema_34_val
        
        if position == 0:
            # Look for entry signals: breakout in direction of 1w trend
            long_signal = (close_1d_val > r1_val) and is_uptrend and vol_spike[i]
            short_signal = (close_1d_val < s1_val) and (not is_uptrend) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_1d_val
                days_since_entry = 0
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_1d_val
                days_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            days_since_entry += 1
            # Exit conditions:
            # 1. Minimum holding period of 3 days to avoid whipsaw
            # 2. Price closes below S1 (opposite Camarilla level)
            # 3. ATR-based stoploss: 2.0 * ATR below entry
            if days_since_entry >= 3:
                exit_signal = close_1d_val < s1_val
                stop_signal = close_1d_val < (entry_price - 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            days_since_entry += 1
            # Exit conditions:
            # 1. Minimum holding period of 3 days to avoid whipsaw
            # 2. Price closes above R1 (opposite Camarilla level)
            # 3. ATR-based stoploss: 2.0 * ATR above entry
            if days_since_entry >= 3:
                exit_signal = close_1d_val > r1_val
                stop_signal = close_1d_val > (entry_price + 2.0 * atr_val)
                if exit_signal or stop_signal:
                    signals[i] = 0.0
                    position = 0
                    days_since_entry = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0