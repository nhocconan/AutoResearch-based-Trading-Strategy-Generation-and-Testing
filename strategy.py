#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_HTFRegime
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike (ATR ratio > 1.5) plus choppiness regime filter (CHOP > 61.8 = ranging). 
Trades only in ranging markets where mean reversion at extreme pivot levels works best. Uses discrete sizing 0.25 to limit trades (~30/year). 
Volume spike ensures institutional participation. Works in bull/bear via trend filter and volatility regime.
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
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
    # Use 6-period lookback for 4h data (6*4h = 24h = 1 day)
    lookback = 6
    prev_high = pd.Series(high).shift(lookback).rolling(window=lookback, min_periods=lookback).max().values
    prev_low = pd.Series(low).shift(lookback).rolling(window=lookback, min_periods=lookback).min().values
    prev_close = pd.Series(close).shift(lookback).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    camarilla_h4 = prev_close + range_val * 1.1 / 6
    camarilla_l4 = prev_close - range_val * 1.1 / 6
    
    # Calculate Choppiness Index (CHOP) for regime filter
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for ATR ratio and EMA, 6 for Camarilla, 14 for CHOP)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        h4_val = camarilla_h4[i]
        l4_val = camarilla_l4[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike
        chop_val = chop[i]
        in_range = chop_val > 61.8  # ranging market regime
        size = fixed_size
        
        # Entry conditions: Camarilla R3/S3 breakout with volume spike AND aligned with 12h EMA50 trend AND in ranging market
        long_entry = (close_val > r3_val) and vol_spike and (close_val > ema_50_val) and in_range
        short_entry = (close_val < s3_val) and vol_spike and (close_val < ema_50_val) and in_range
        
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
            if close_val < h4_val and close_val > l4_val:  # back inside H4/L4
                signals[i] = 0.0
                position = 0
            elif close_val < ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla H4/L4 or trend reversal
            if close_val > l4_val and close_val < h4_val:  # back inside H4/L4
                signals[i] = 0.0
                position = 0
            elif close_val > ema_50_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_HTFRegime"
timeframe = "4h"
leverage = 1.0