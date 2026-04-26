#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA34 trend filter and volume spike (ATR ratio > 1.5). Trade only breakouts aligned with 1w trend during volatility expansion. Uses discrete sizing 0.25 to limit trades (~15/year). Volume spike ensures institutional participation. Works in bull/bear via trend filter and volatility regime.
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
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
    lookback = 1  # 1 day lookback for daily data
    prev_high = pd.Series(high).shift(lookback).rolling(window=lookback, min_periods=lookback).max().values
    prev_low = pd.Series(low).shift(lookback).rolling(window=lookback, min_periods=lookback).min().values
    prev_close = pd.Series(close).shift(lookback).rolling(window=lookback, min_periods=lookback).mean().values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r1 = prev_close + range_val * 1.1 / 12
    camarilla_s1 = prev_close - range_val * 1.1 / 12
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 50 for ATR ratio, 1 for Camarilla)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]) or
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = camarilla_r1[i]
        s1_val = camarilla_s1[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume spike
        size = fixed_size
        
        # Entry conditions: Camarilla breakout with volume spike AND aligned with 1w EMA34 trend
        long_entry = (close_val > r1_val) and vol_spike and (close_val > ema_34_val)
        short_entry = (close_val < s1_val) and vol_spike and (close_val < ema_34_val)
        
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
            camarilla_h4 = prev_close + range_val * 1.1 / 6
            camarilla_l4 = prev_close - range_val * 1.1 / 6
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
            camarilla_h4 = prev_close + range_val * 1.1 / 6
            camarilla_l4 = prev_close - range_val * 1.1 / 6
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

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0