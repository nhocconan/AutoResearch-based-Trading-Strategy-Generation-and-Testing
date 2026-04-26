#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA20 trend filter and volume confirmation (>1.8x average) during active session (08-20 UTC). 
Designed for both bull and bear markets: 4h EMA20 captures medium-term trend, volume ensures conviction, session filter avoids low-liquidity periods.
Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA20 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # ATR(14) on 1h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Camarilla R1 and S1 from prior 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    if len(high_4h) < 2:
        camarilla_r1 = np.full_like(close_4h_arr, np.nan)
        camarilla_s1 = np.full_like(close_4h_arr, np.nan)
    else:
        camarilla_r1 = close_4h_arr[:-1] + 1.1 * (high_4h[:-1] - low_4h[:-1]) / 12
        camarilla_s1 = close_4h_arr[:-1] - 1.1 * (high_4h[:-1] - low_4h[:-1]) / 12
        camarilla_r1 = np.concatenate([[np.nan], camarilla_r1])
        camarilla_s1 = np.concatenate([[np.nan], camarilla_s1])
    
    # Align Camarilla levels to 1h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of volume MA (20), 4h EMA (20), ATR (14)
    start_idx = max(20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_20_4h_val = ema_20_4h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = vol_val > 1.8 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 1.5*ATR
        breakout_threshold = 1.5 * atr_val
        
        if position == 0:
            # Long: close above R1 + threshold, uptrend (close > EMA20_4h), volume confirmation
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_20_4h_val) and volume_confirmed
            # Short: close below S1 - threshold, downtrend (close < EMA20_4h), volume confirmation
            short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_20_4h_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price closes below S1
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA20_4h)
            elif close_val < ema_20_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price closes above R1
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA20_4h)
            elif close_val > ema_20_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0