#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, use 4h Camarilla R1/S1 breakouts aligned with 4h EMA50 trend filter and volume spike (>2.0x 20-period average) for high-conviction entries. Apply UTC 8-20 session filter to avoid low-liquidity periods. Discrete sizing 0.20 to minimize fee drag. Target 15-30 trades/year by requiring confluence of breakout, trend, volume, and session. Works in bull/bear via 4h trend alignment and volume confirmation to filter false breakouts.
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
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # ATR(14) on 4h for breakout confirmation and volatility filter
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h_arr[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h_arr[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Camarilla R1 and S1 from prior 4h bar
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
    
    # Volume average (20-period) for confirmation on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of volume MA (20), 4h EMA (50), 4h ATR (14)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr_4h_aligned[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        atr_val = atr_4h_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average (strict)
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        # Breakout threshold: price must close beyond Camarilla level by 2.0*ATR (strict)
        breakout_threshold = 2.0 * atr_val
        
        if position == 0:
            # Long: close above R1 + threshold, uptrend (close > EMA50_4h), volume confirmation, in session
            long_signal = (close_val > r1_val + breakout_threshold) and (close_val > ema_50_4h_val) and volume_confirmed
            # Short: close below S1 - threshold, downtrend (close < EMA50_4h), volume confirmation, in session
            short_signal = (close_val < s1_val - breakout_threshold) and (close_val < ema_50_4h_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below EMA50_4h (trend reversal)
            if close_val < ema_50_4h_val:
                signals[i] = 0.0
                position = 0
            # Exit: price closes below S1 (mean reversion)
            elif close_val < s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above EMA50_4h (trend reversal)
            if close_val > ema_50_4h_val:
                signals[i] = 0.0
                position = 0
            # Exit: price closes above R1 (mean reversion)
            elif close_val > r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0