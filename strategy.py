#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakouts aligned with 12h trend, volume confirmation, and choppiness regime filter capture sustained moves while avoiding whipsaws in ranging markets. Uses chop > 61.8 for ranging (mean-reversion exit) and chop < 38.2 for trending (breakout entry). Discrete sizing (0.25) limits fee churn. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 12h and 1d data
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 12h bar (for 4h chart)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    rng_12h = high_12h - low_12h
    camarilla_r3 = close_12h + 1.1 * rng_12h / 2
    camarilla_s3 = close_12h - 1.1 * rng_12h / 2
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA34 for additional trend confirmation
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (4h)
    r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h equivalent)
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
    
    # Warmup: need 12h EMA50 (50), volume avg (24), chop (14)
    start_idx = max(50, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        ema_1d_val = ema_34_1d_aligned[i]
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
            # Determine combined trend: both 12h and 1d must agree
            is_uptrend = (close_val > ema_12h_val) and (close_val > ema_1d_val)
            is_downtrend = (close_val < ema_12h_val) and (close_val < ema_1d_val)
            
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
            exit_condition = (close_val < s3_val) or (close_val < ema_12h_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend changes to uptrend
            exit_condition = (close_val > r3_val) or (close_val > ema_12h_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0