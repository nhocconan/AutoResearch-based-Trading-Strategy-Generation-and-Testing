#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike (ATR ratio > 1.5), and choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend). 
In trending regimes (CHOP < 38.2): trade breakouts aligned with 1d EMA34 trend. 
In ranging regimes (CHOP > 61.8): trade mean reversion at Camarilla H4/L4 levels.
Volume spike ensures institutional participation. Discrete sizing 0.25 limits trades (~30/year) to reduce fee drag.
Works in bull/bear via trend filter and volatility regime adaptation.
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
    
    # Calculate Choppiness Index (CHOP) for regime filter
    chop_period = 14
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / np.log10(chop_period) / (hh - ll))
    
    # Calculate previous day's high/low/close for Camarilla levels
    # Use 6-period lookback for 4h data (6*4h = 24h = 1 day)
    lookback = 6
    prev_high = pd.Series(high).shift(lookback).rolling(window=lookback, min_periods=lookback).max().values
    prev_low = pd.Series(low).shift(lookback).rolling(window=lookback, min_periods=lookback).min().values
    prev_close = pd.Series(close).shift(lookback).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12
    camarilla_h4 = prev_close + range_val * 1.1 / 6
    camarilla_l4 = prev_close - range_val * 1.1 / 6
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for ATR ratio, 34 for EMA, 14 for CHOP, 6 for Camarilla)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = camarilla_r1[i]
        s1_val = camarilla_s1[i]
        h4_val = camarilla_h4[i]
        l4_val = camarilla_l4[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike
        chop_val = chop[i]
        
        # Regime determination
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        
        if position == 0:
            # Flat - look for entry
            if is_trending and vol_spike:
                # In trending regime: trade breakouts aligned with 1d EMA34 trend
                long_entry = (close_val > r1_val) and (close_val > ema_34_val)
                short_entry = (close_val < s1_val) and (close_val < ema_34_val)
                if long_entry:
                    signals[i] = fixed_size
                    position = 1
                elif short_entry:
                    signals[i] = -fixed_size
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # In ranging regime: trade mean reversion at H4/L4 levels
                long_entry = (close_val < l4_val) and (close_val > prev_close[i])  # bounce from L4
                short_entry = (close_val > h4_val) and (close_val < prev_close[i])  # rejection at H4
                if long_entry:
                    signals[i] = fixed_size
                    position = 1
                elif short_entry:
                    signals[i] = -fixed_size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # In transition regime: no trades
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            if is_trending:
                # In trending regime: exit on trend reversal or re-entry into R1/S1 zone
                if close_val < ema_34_val or (close_val < r1_val and close_val > s1_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = fixed_size
            else:
                # In ranging/transition regime: exit at mean (prev_close) or opposite H4/L4
                if close_val >= prev_close[i] or close_val <= l4_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = fixed_size
        elif position == -1:
            # Short - exit conditions
            if is_trending:
                # In trending regime: exit on trend reversal or re-entry into R1/S1 zone
                if close_val > ema_34_val or (close_val > s1_val and close_val < r1_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -fixed_size
            else:
                # In ranging/transition regime: exit at mean (prev_close) or opposite H4/L4
                if close_val <= prev_close[i] or close_val >= h4_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -fixed_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0