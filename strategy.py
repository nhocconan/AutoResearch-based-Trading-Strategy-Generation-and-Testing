#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_ChopRegime
Hypothesis: Camarilla R3/S3 breakouts aligned with 1d EMA34 trend, volume confirmation, and chop regime filter capture sustained moves while avoiding sideways whipsaws. The chop filter (Bollinger Band Width percentile) ensures trades only occur in trending markets, improving performance in both bull and bear cycles. Discrete sizing (0.25) limits fee churn.
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
    
    # Get 1d data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    rng_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng_1d / 2
    camarilla_s3 = close_1d - 1.1 * rng_1d / 2
    
    # Align all indicators to primary timeframe (4h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: Bollinger Band Width percentile < 50% (trending market)
    bb_window = 20
    bb_std = 2.0
    bb_ma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = bb_ma + bb_std * bb_std_dev
    bb_lower = bb_ma - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    bb_width_std = pd.Series(bb_width).rolling(window=50, min_periods=50).std().values
    # Avoid division by zero
    bb_width_z = np.where(bb_width_std > 0, (bb_width - bb_width_ma) / bb_width_std, 0)
    # Convert z-score to percentile approximation (CDF of normal)
    chop_value = 50 * (1 + np.erf(bb_width_z / np.sqrt(2)))
    chop_value = np.clip(chop_value, 0, 100)
    # Trending market: chop < 50 (lower BB width percentile)
    trending_regime = chop_value < 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA34 (34), volume avg (24), BB width (50)
    start_idx = max(34, 24, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_conf = volume_confirm[i]
        is_trending = trending_regime[i]
        
        if position == 0:
            # Determine trend: price relative to 1d EMA34
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend and is_trending:
                # Uptrend: long when price breaks above R3 and volume confirms
                if (close_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend and is_trending:
                # Downtrend: short when price breaks below S3 and volume confirms
                if (close_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S3 (support) or trend changes to downtrend
            exit_condition = (close_val < s3_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R3 (resistance) or trend changes to uptrend
            exit_condition = (close_val > r3_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_ChopRegime"
timeframe = "4h"
leverage = 1.0