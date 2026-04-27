#!/usr/bin/env python3
"""
6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_HTFRegime
Hypothesis: Camarilla R4/S4 breakouts on 6h timeframe with 1d trend filter (EMA50), volume spike confirmation (>2.0x 24-period average), and HTF regime filter (weekly ADX > 25) capture strong momentum moves while avoiding choppy markets. Designed for 6h timeframe with tight entries (target: 50-150 trades over 4 years) to minimize fee drag. Works in both bull and bear regimes by following the higher timeframe trend.
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R4, S4 levels: R4 = close + 1.1*(high-low), S4 = close - 1.1*(high-low)
    rng_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * rng_1d
    camarilla_s4 = close_1d - 1.1 * rng_1d
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for HTF regime filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.append([np.nan], close_1w[:-1]))
    tr3 = np.abs(low_1w - np.append([np.nan], close_1w[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.append([np.nan], high_1w[:-1])) > (np.append([np.nan], low_1w[:-1]) - low_1w),
                       np.maximum(high_1w - np.append([np.nan], high_1w[:-1]), 0), 0)
    dm_minus = np.where((np.append([np.nan], low_1w[:-1]) - low_1w) > (high_1w - np.append([np.nan], high_1w[:-1])),
                        np.maximum(np.append([np.nan], low_1w[:-1]) - low_1w, 0), 0)
    
    # Smoothed values
    tr_period = 14
    tr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    di_minus = 100 * dm_minus_smooth / np.where(tr_smooth == 0, 1, tr_smooth)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align all indicators to primary timeframe (6h)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA50 (50), 1w ADX (14+14=28), volume avg (24)
    start_idx = max(50, 28, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        ema_1d_val = ema_50_1d_aligned[i]
        adx_val = adx_aligned[i]
        vol_conf = volume_confirm[i]
        
        # Only trade when HTF regime is trending (ADX > 25)
        is_trending_regime = adx_val > 25
        
        if not is_trending_regime:
            # In ranging markets, exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend: price relative to 1d EMA50
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R4 and volume confirms
                if (close_val > r4_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S4 and volume confirms
                if (close_val < s4_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S4 (support) or trend changes to downtrend
            exit_condition = (close_val < s4_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R4 (resistance) or trend changes to uptrend
            exit_condition = (close_val > r4_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R4_S4_Breakout_1dTrend_VolumeSpike_HTFRegime"
timeframe = "6h"
leverage = 1.0