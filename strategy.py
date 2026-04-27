#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_ChopRegime
Hypothesis: On daily timeframe, Camarilla R3/S3 breakouts aligned with weekly trend, volume confirmation, and choppiness regime filter capture sustained moves while avoiding whipsaws. Uses chop < 38.2 for trending regime (entry allowed) and chop > 61.8 for ranging (exit). Discrete sizing (0.25) limits fee churn. Target: 30-100 total trades over 4 years.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla R3, S3 levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    rng_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng_1d / 2
    camarilla_s3 = close_1d - 1.1 * rng_1d / 2
    
    # Align all indicators to primary timeframe (1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (equivalent to 24 * 1h = 1d)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter (14-period) - chop > 61.8 = ranging, chop < 38.2 = trending
    # We only trade in trending regimes (chop < 38.2) to avoid whipsaws
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    true_range = pd.Series(np.maximum(high - low, 
                                     np.maximum(np.abs(high - np.append([np.nan], close[:-1])),
                                                np.abs(low - np.append([np.nan], close[:-1]))))).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_filter = chop < 38.2  # Only trade when market is trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly EMA34 (34), volume avg (24), chop (14)
    start_idx = max(34, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_1w_val = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        is_trending = chop_filter[i]
        
        # Only enter trades in trending regime
        if not is_trending:
            # In ranging markets, exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend: price relative to weekly EMA34
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R3 and volume confirms
                if (close_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S3 and volume confirms
                if (close_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S3 (support) or trend changes to downtrend
            exit_condition = (close_val < s3_val) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend changes to uptrend
            exit_condition = (close_val > r3_val) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_ChopRegime"
timeframe = "1d"
leverage = 1.0