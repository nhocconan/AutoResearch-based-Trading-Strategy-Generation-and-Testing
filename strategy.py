#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike (ATR ratio > 1.5). 
Camarilla R3/S3 represent stronger support/resistance than R1/S1, reducing false breakouts. 
Volume spike confirms institutional participation. 1d EMA34 filter ensures alignment with daily trend. 
Discrete sizing 0.25 limits trade frequency (~20-30/year). Works in bull/bear via trend filter.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate previous day's high/low/close for Camarilla levels
    # For 6h data, 4 periods = 1 day (4 * 6h = 24h)
    lookback = 4
    prev_high = pd.Series(high).shift(lookback).rolling(window=lookback, min_periods=lookback).max().values
    prev_low = pd.Series(low).shift(lookback).rolling(window=lookback, min_periods=lookback).min().values
    prev_close = pd.Series(close).shift(lookback).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 50 for ATR ratio, 4 for Camarilla)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike threshold
        size = fixed_size
        
        # Entry conditions: Camarilla breakout with volume spike AND aligned with 1d EMA34 trend
        long_entry = (close_val > r3_val) and vol_spike and (close_val > ema_34_val)
        short_entry = (close_val < s3_val) and vol_spike and (close_val < ema_34_val)
        
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
            # Long - exit on Camarilla H4/L4 or trend reversal
            camarilla_h4 = prev_close + range_val * 1.1 / 2
            camarilla_l4 = prev_close - range_val * 1.1 / 2
            h4_val = camarilla_h4[i]
            l4_val = camarilla_l4[i]
            if close_val < h4_val and close_val > l4_val:  # back inside H4/L4
                signals[i] = 0.0
                position = 0
            elif close_val < ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla H4/L4 or trend reversal
            camarilla_h4 = prev_close + range_val * 1.1 / 2
            camarilla_l4 = prev_close - range_val * 1.1 / 2
            h4_val = camarilla_h4[i]
            l4_val = camarilla_l4[i]
            if close_val > l4_val and close_val < h4_val:  # back inside H4/L4
                signals[i] = 0.0
                position = 0
            elif close_val > ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0