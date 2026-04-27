#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Camarilla R3/S3 breakouts on 6h timeframe, aligned with 12h EMA34 trend and volume confirmation, capture high-probability moves. 
Uses 12h HTF for trend and volume context to avoid whipsaws. Discrete sizing (0.25) balances return and fee drag. 
Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via trend filter and volume confirmation.
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
    
    # Get 6h data for Camarilla levels (prior 6h bar)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    range_6h = high_6h - low_6h
    camarilla_r3 = close_6h + 1.125 * range_6h
    camarilla_s3 = close_6h - 1.125 * range_6h
    
    # Get 12h data for EMA34 trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA34 for trend
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 12h volume average (20-period) for confirmation
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h primary timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s3)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Camarilla (1), EMA34 (34), volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_avg_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_current = volume[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema34 = ema34_12h_aligned[i]
        vol_avg = vol_avg_12h_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5 * 12h volume average
        volume_confirm = vol_current > (1.5 * vol_avg)
        
        if position == 0:
            # Determine trend: price vs 12h EMA34
            uptrend = close_val > ema34
            downtrend = close_val < ema34
            
            if uptrend and volume_confirm:
                # Long: break above R3 with volume in uptrend
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and volume_confirm:
                # Short: break below S3 with volume in downtrend
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long exit: stoploss (2.0*ATR) or Camarilla S3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price - 2.0 * atr_approx
            
            if close_val <= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val < s3:  # Camarilla S3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: stoploss (2.0*ATR) or Camarilla R3 touch
            atr_approx = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values[i]
            stop_loss = entry_price + 2.0 * atr_approx
            
            if close_val >= stop_loss:
                signals[i] = 0.0
                position = 0
            elif close_val > r3:  # Camarilla R3 touch
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0