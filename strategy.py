#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (ATR ratio > 1.5), and chop regime filter (CHOP > 61.8). Targets 20-40 trades/year by requiring confluence of trend, volatility expansion, and ranging market. Works in bull/bear via 1d trend filter and volatility regime. Uses discrete sizing 0.25 to balance return and drawdown.
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
    
    # Calculate ATR(14) for volume regime and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Choppiness Index (CHOP) for regime filter
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum()
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max()
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(chop_period)
    chop_values = chop.values
    
    # Calculate previous day's high/low/close for Camarilla levels
    # Use 6-period lookback for 4h data (6*4h = 1 day)
    lookback = 6
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
    
    # Warmup: max of calculations (50 for ATR ratio, 34 for EMA, 14 for CHOP, 6 for Camarilla)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike
        chop_val = chop_values[i]
        in_chop_regime = chop_val > 61.8  # ranging market
        size = fixed_size
        
        # Entry conditions: Camarilla breakout with volume spike AND aligned with 1d EMA34 trend AND in chop regime
        long_entry = (close_val > r3_val) and vol_spike and (close_val > ema_34_val) and in_chop_regime
        short_entry = (close_val < s3_val) and vol_spike and (close_val < ema_34_val) and in_chop_regime
        
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
            # Long - exit on Camarilla H6/L6 or trend reversal
            camarilla_h6 = prev_close + range_val * 1.1 / 2
            camarilla_l6 = prev_close - range_val * 1.1 / 2
            h6_val = camarilla_h6[i]
            l6_val = camarilla_l6[i]
            if close_val < h6_val and close_val > l6_val:  # back inside H6/L6
                signals[i] = 0.0
                position = 0
            elif close_val < ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Camarilla H6/L6 or trend reversal
            camarilla_h6 = prev_close + range_val * 1.1 / 2
            camarilla_l6 = prev_close - range_val * 1.1 / 2
            h6_val = camarilla_h6[i]
            l6_val = camarilla_l6[i]
            if close_val > l6_val and close_val < h6_val:  # back inside H6/L6
                signals[i] = 0.0
                position = 0
            elif close_val > ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0