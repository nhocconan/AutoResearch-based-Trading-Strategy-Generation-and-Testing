#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On 12h timeframe, trade Camarilla R3/S3 breakouts aligned with 1-week EMA50 trend filter and volume spike (ATR ratio > 1.3). Uses discrete sizing 0.25 to limit trades (~15-30/year). Weekly trend ensures alignment with major market bias, reducing false breakouts in chop. Works in bull/bear via weekly EMA50 slope and volume confirmation.
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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for volume regime and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    range_prev = high_prev - low_prev
    
    camarilla_r3 = close_prev + range_prev * 1.1 / 4
    camarilla_s3 = close_prev - range_prev * 1.1 / 4
    
    # Align HTF arrays to 12h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio and EMA, plus 1 for Camarilla shift
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = atr_ratio[i] > 1.3
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry aligned with weekly trend
            long_entry = (close_val > camarilla_r3_val) and vol_spike and (close_val > ema_50_val)
            short_entry = (close_val < camarilla_s3_val) and vol_spike and (close_val < ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or re-entry to Camarilla H3-L3
            # Use 1d H3/L3 for exit (mean reversion levels)
            camarilla_h3 = close_prev[i] + range_prev[i] * 1.1 / 2 if not np.isnan(close_prev[i]) and not np.isnan(range_prev[i]) else np.nan
            camarilla_l3 = close_prev[i] - range_prev[i] * 1.1 / 2 if not np.isnan(close_prev[i]) and not np.isnan(range_prev[i]) else np.nan
            
            if np.isnan(camarilla_h3) or np.isnan(camarilla_l3):
                signals[i] = size
                continue
                
            if close_val < camarilla_h3 and close_val > camarilla_l3:
                signals[i] = 0.0
                position = 0
            elif close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or re-entry to Camarilla H3-L3
            camarilla_h3 = close_prev[i] + range_prev[i] * 1.1 / 2 if not np.isnan(close_prev[i]) and not np.isnan(range_prev[i]) else np.nan
            camarilla_l3 = close_prev[i] - range_prev[i] * 1.1 / 2 if not np.isnan(close_prev[i]) and not np.isnan(range_prev[i]) else np.nan
            
            if np.isnan(camarilla_h3) or np.isnan(camarilla_l3):
                signals[i] = -size
                continue
                
            if close_val > camarilla_l3 and close_val < camarilla_h3:
                signals[i] = 0.0
                position = 0
            elif close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0