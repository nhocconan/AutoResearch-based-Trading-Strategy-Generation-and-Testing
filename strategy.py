#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 1d with 1w EMA50 trend filter and volume spike confirmation.
Uses 1w EMA50 for slower, more stable trend adaptation suitable for daily timeframe.
Volume confirmation requires 2.0x 20-period median volume to avoid false breakouts.
Only trade in trending markets (weekly ADX > 25) to avoid whipsaws in ranging regimes.
Designed for 15-25 trades/year on BTC/ETH/SOL, working in both bull and bear markets by following weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter (slower adaptation for daily timeframe)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1w OHLC
    prev_close_1w = df_1w['close'].shift(1).values
    prev_high_1w = df_1w['high'].shift(1).values
    prev_low_1w = df_1w['low'].shift(1).values
    
    camarilla_r1 = prev_close_1w + 1.125 * (prev_high_1w - prev_low_1w)
    camarilla_s1 = prev_close_1w - 1.125 * (prev_high_1w - prev_low_1w)
    
    # Align HTF indicators to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume confirmation: 2.0x median volume (20-period) for breakout validation
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ADX(14) on weekly data for regime filter - trending when > 25
    # Calculate ADX using 1w data
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    wn = len(wh)
    
    plus_dm = np.zeros(wn)
    minus_dm = np.zeros(wn)
    tr = np.zeros(wn)
    
    for i in range(1, wn):
        plus_dm[i] = max(wh[i] - wh[i-1], 0) if (wh[i] - wh[i-1]) > (wl[i-1] - wl[i]) else 0
        minus_dm[i] = max(wl[i-1] - wl[i], 0) if (wl[i-1] - wl[i]) > (wh[i] - wh[i-1]) else 0
        tr[i] = max(wh[i] - wl[i], abs(wh[i] - wc[i-1]), abs(wl[i] - wc[i-1]))
    
    # Wilder's smoothing
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if wn >= period:
        plus_dm_smooth = WilderSmooth(plus_dm, period)
        minus_dm_smooth = WilderSmooth(minus_dm, period)
        tr_smooth = WilderSmooth(tr, period)
        
        # Avoid division by zero
        plus_di = 100 * plus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        minus_di = 100 * minus_dm_smooth / np.where(tr_smooth != 0, tr_smooth, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1)
        adx_1w = WilderSmooth(dx, period)
    else:
        adx_1w = np.full(wn, np.nan)
    
    # Align weekly ADX to daily timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1w EMA (50), volume median (20), ADX (14*2 for smoothing)
    start_idx = max(50, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        adx_val = adx_1w_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend, and trending regime
            long_signal = (close_val > camarilla_r1_aligned[i]) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (close_val > ema_50_1w_val) and \
                          (adx_val > 25)
            
            # Short: break below S1 with volume spike, downtrend, and trending regime
            short_signal = (close_val < camarilla_s1_aligned[i]) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (close_val < ema_50_1w_val) and \
                           (adx_val > 25)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) or trend changes (close < 1w EMA50) or regime changes (ADX < 20)
            if (close_val < camarilla_s1_aligned[i]) or \
               (close_val < ema_50_1w_val) or \
               (adx_val < 20):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) or trend changes (close > 1w EMA50) or regime changes (ADX < 20)
            if (close_val > camarilla_r1_aligned[i]) or \
               (close_val > ema_50_1w_val) or \
               (adx_val < 20):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0