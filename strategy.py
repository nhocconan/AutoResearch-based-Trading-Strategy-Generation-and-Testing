#!/usr/bin/env python3
# 1h_1H_Camarilla_R2_S2_Breakout_4hTrend_1dVolume
# Hypothesis: For 1h timeframe, use 4h trend (EMA50) and 1d volume spike to filter Camarilla R2/S2 breakouts.
# This reduces false signals by aligning with higher timeframe trend and volume confirmation.
# Works in bull/bear: 4h EMA50 filter avoids counter-trend trades, 1d volume spike confirms breakout strength.
# Target: 15-37 trades/year by using tight entry conditions and session filter (08-20 UTC).

name = "1h_1H_Camarilla_R2_S2_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mta_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    # Calculate Camarilla levels (R2, S2)
    rang = ph - pl
    r2 = pc + 1.1 * rang * 1.0833
    s2 = pc - 1.1 * rang * 1.0833
    
    # Align Camarilla levels to 1h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 1d volume 20-period average for spike detection
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma_1d[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma_1d[i] = (vol_ma_1d[i-1] * 19 + volume_1d[i]) / 20
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume spike: current 1h volume vs 1d average volume scaled to 1h
    # Approximate: 1d volume / 24 = average hourly volume
    vol_ma_1h_equiv = vol_ma_1d_aligned / 24.0
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma_1h_equiv)) & (vol_ma_1h_equiv != 0)
    volume_ratio[valid] = volume[valid] / vol_ma_1h_equiv[valid]
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 AND uptrend (price > 4h EMA50) AND volume spike
            if (close[i] > r2_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S2 AND downtrend (price < 4h EMA50) AND volume spike
            elif (close[i] < s2_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 OR trend reversal (price < 4h EMA50)
            if close[i] < s2_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R2 OR trend reversal (price > 4h EMA50)
            if close[i] > r2_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals