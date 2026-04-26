#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation (ATR ratio > 1.2). Camarilla levels from 1d provide key support/resistance; breakout above R3 or below S3 with volume and 1d trend alignment captures strong momentum moves. Discrete sizing 0.25 limits trades to ~12-37/year. Works in bull/bear via 1d trend filter. Uses 12h primary timeframe as requested.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    # Camarilla: based on previous day's range
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data to compute today's levels
    range_prev = high_prev - low_prev
    camarilla_r3 = close_prev + range_prev * 1.1 / 4
    camarilla_s3 = close_prev - range_prev * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (1d -> 12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio and EMA alignment, plus 1 for Camarilla shift
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.2  # volume regime
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        size = fixed_size
        
        # Entry conditions: Camarilla R3/S3 breakout with volume spike AND aligned with 1d EMA50 trend
        # Long: price breaks above R3 (bullish breakout)
        # Short: price breaks below S3 (bearish breakout)
        long_entry = (close_val > camarilla_r3_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_s3_val) and vol_spike and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters Camarilla H3-L3 range or trend reversal
            # H3 = close_prev + range_prev * 1.1/2, L3 = close_prev - range_prev * 1.1/2
            camarilla_h3 = close_prev + range_prev * 1.1 / 2
            camarilla_l3 = close_prev - range_prev * 1.1 / 2
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            h3_val = camarilla_h3_aligned[i]
            l3_val = camarilla_l3_aligned[i]
            
            if (close_val < h3_val and close_val > l3_val) or close_val < ema_50_val:  # back inside H3-L3 or trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla H3-L3 range or trend reversal
            camarilla_h3 = close_prev + range_prev * 1.1 / 2
            camarilla_l3 = close_prev - range_prev * 1.1 / 2
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            h3_val = camarilla_h3_aligned[i]
            l3_val = camarilla_l3_aligned[i]
            
            if (close_val > l3_val and close_val < h3_val) or close_val > ema_50_val:  # back inside H3-L3 or trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0