#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Use 1h timeframe with Camarilla R3/S3 breakout confirmed by 4h EMA50 trend and volume spike. Targets 15-37 trades/year to minimize fee drag. Works in bull/bear markets by using 4h EMA50 for trend direction and volume confirmation to filter false breakouts. Includes ATR-based stoploss to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (R3, S3) on 1h
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    hl_range = high - low
    camarilla_r3 = close + 1.1 * hl_range / 2
    camarilla_s3 = close - 1.1 * hl_range / 2
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for volume avg, 50 for 4h EMA, 14 for ATR
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.20  # 20% position size
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Flat - look for breakout with trend and volume confirmation
            # Long: break above Camarilla R3 + 4h EMA50 uptrend + volume spike
            long_entry = (close_val > camarilla_r3[i]) and \
                       (ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1]) and \
                       volume_spike[i]
            # Short: break below Camarilla S3 + 4h EMA50 downtrend + volume spike
            short_entry = (close_val < camarilla_s3[i]) and \
                        (ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Camarilla S3 break or ATR stoploss
            exit_condition = (close_val < camarilla_s3[i]) or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla R3 break or ATR stoploss
            exit_condition = (close_val > camarilla_r3[i]) or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0