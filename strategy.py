#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_HTF_Filter
Hypothesis: Camarilla R3/S3 breakouts aligned with 1d EMA34 trend and volume spike capture sustained moves. Adding a 12h HTF regime filter (EMA50 slope) improves edge by avoiding counter-trend breakouts in choppy markets. ATR-based stoploss (2.0x ATR) controls drawdown. Discrete sizing (0.25) limits fee churn. Target: 75-150 trades over 4 years.
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
    
    # Get 12h data for HTF regime filter (EMA50 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    # Calculate slope of EMA50 over 3 bars (~1.5 days) to determine regime
    ema50_slope = (ema_50_12h - np.roll(ema_50_12h, 3)) / 3
    ema50_slope[0:3] = 0  # first 3 values invalid
    
    # Align all indicators to primary timeframe (4h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    # Volume confirmation: current volume > 1.8 * 24-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # ATR for dynamic stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA34 (34), 12h EMA50 slope (50+3), volume avg (24), ATR (14)
    start_idx = max(53, 34, 24, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema50_slope_val = ema50_slope_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Determine trend: price relative to 1d EMA34
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            # Only take breakouts in direction of 12h EMA50 slope (regime filter)
            if is_uptrend and (ema50_slope_val > 0):
                # Uptrend regime: long when price breaks above R3, volume confirms
                if (close_val > r3_val) and vol_conf:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif is_downtrend and (ema50_slope_val < 0):
                # Downtrend regime: short when price breaks below S3, volume confirms
                if (close_val < s3_val) and vol_conf:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit conditions: stoploss, trend reversal, or S3 touch
            stop_loss = entry_price - 2.0 * atr_val
            trend_reversal = close_val < ema_1d_val
            support_touch = close_val < s3_val
            
            if stop_loss > 0 and close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif trend_reversal or support_touch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: stoploss, trend reversal, or R3 touch
            stop_loss = entry_price + 2.0 * atr_val
            trend_reversal = close_val > ema_1d_val
            resistance_touch = close_val > r3_val
            
            if stop_loss > 0 and close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif trend_reversal or resistance_touch:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_HTF_Filter"
timeframe = "4h"
leverage = 1.0