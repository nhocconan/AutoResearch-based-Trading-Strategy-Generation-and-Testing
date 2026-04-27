#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_ChopRegime
Hypothesis: Daily Camarilla R3/S3 breakouts with weekly trend alignment, volume confirmation, and chop regime filter capture sustained moves while avoiding whipsaws. Weekly EMA50 defines trend, chop filter (EWMA-based) avoids range-bound false breakouts. Discrete sizing (0.25) limits fee churn. Target: 30-100 trades over 4 years (7-25/year).
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
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA34 for additional trend confirmation (optional)
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate chop regime filter using EWMA of True Range (simpler than ATR)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = 0
    atr_ewma = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Chop filter: low volatility regime (avoid breakouts in high volatility)
    atr_ratio = atr_ewma / pd.Series(atr_ewma).rolling(window=50, min_periods=50).mean().values
    chop_filter = atr_ratio < 1.2  # Only trade when volatility is below 1.2x 50-period average
    
    # Calculate Camarilla levels from previous daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    rng_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * rng_1d / 2
    camarilla_s3 = close_1d - 1.1 * rng_1d / 2
    
    # Align all indicators to primary timeframe (1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need weekly EMA50 (50), daily EMA34 (34), volume avg (20), chop filter (50)
    start_idx = max(50, 34, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(chop_filter_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        weekly_ema_val = ema_50_1w_aligned[i]
        daily_ema_val = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter_aligned[i]
        
        if position == 0:
            # Determine trend: price above/both EMAs for uptrend, below/both for downtrend
            is_uptrend = close_val > weekly_ema_val and close_val > daily_ema_val
            is_downtrend = close_val < weekly_ema_val and close_val < daily_ema_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R3, volume confirms, and chop filter OK
                if (close_val > r3_val) and vol_conf and chop_ok:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif is_downtrend:
                # Downtrend: short when price breaks below S3, volume confirms, and chop filter OK
                if (close_val < s3_val) and vol_conf and chop_ok:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: trend reversal or S3 touch (support)
            trend_reversal = close_val < weekly_ema_val or close_val < daily_ema_val
            support_touch = close_val < s3_val
            
            if trend_reversal or support_touch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: trend reversal or R3 touch (resistance)
            trend_reversal = close_val > weekly_ema_val or close_val > daily_ema_val
            resistance_touch = close_val > r3_val
            
            if trend_reversal or resistance_touch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_ChopRegime"
timeframe = "1d"
leverage = 1.0