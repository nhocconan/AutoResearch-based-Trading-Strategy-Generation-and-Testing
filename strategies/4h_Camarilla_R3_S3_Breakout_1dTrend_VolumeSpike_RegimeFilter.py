#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume spike (ATR ratio > 1.2), and choppiness regime filter (CHOP > 61.8 = range -> mean reversion at H3/L3; CHOP < 38.2 = trend -> breakout). Uses discrete sizing 0.25 to limit trades (~20-40/year). Works in bull/bear via 1d trend and regime adaptation.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume regime and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate 14-period Choppiness Index for regime filter
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(chop_period)
    
    # Calculate Camarilla levels from previous day (using 1d OHLC)
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    range_prev = high_prev - low_prev
    
    camarilla_r3 = close_prev + range_prev * 1.1 / 4
    camarilla_s3 = close_prev - range_prev * 1.1 / 4
    camarilla_h3 = close_prev + range_prev * 1.1 / 2  # H3/L3 for exits
    camarilla_l3 = close_prev - range_prev * 1.1 / 2
    
    # Align all HTF arrays to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 50 for ATR ratio, 14 for CHOP, plus 1 for Camarilla shift
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or
            np.isnan(chop[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.2
        chop_val = chop[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        camarilla_h3_val = camarilla_h3_aligned[i]
        camarilla_l3_val = camarilla_l3_aligned[i]
        size = fixed_size
        
        # Determine regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (follow breakout)
        in_range = chop_val > 61.8
        in_trend = chop_val < 38.2
        
        if position == 0:
            # Flat - look for entry
            if in_trend:
                # Trend regime: follow breakout
                long_entry = (close_val > camarilla_r3_val) and vol_spike and (close_val > ema_50_val)
                short_entry = (close_val < camarilla_s3_val) and vol_spike and (close_val < ema_50_val)
            else:
                # Range regime: mean reversion at H3/L3
                long_entry = (close_val < camarilla_l3_val) and vol_spike and (close_val < ema_50_val)
                short_entry = (close_val > camarilla_h3_val) and vol_spike and (close_val > ema_50_val)
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            if in_trend:
                # Trend regime: exit on re-entry to H3-L3 or trend reversal
                if (close_val < camarilla_h3_val and close_val > camarilla_l3_val) or close_val < ema_50_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                # Range regime: exit at opposite extreme (H3) or stoploss
                if close_val > camarilla_h3_val or close_val < ema_50_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short - exit conditions
            if in_trend:
                # Trend regime: exit on re-entry to H3-L3 or trend reversal
                if (close_val > camarilla_l3_val and close_val < camarilla_h3_val) or close_val > ema_50_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                # Range regime: exit at opposite extreme (L3) or stoploss
                if close_val < camarilla_l3_val or close_val > ema_50_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0