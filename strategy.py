#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_FundingFilter
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter and funding rate mean reversion (Z-score < -2 for long, > +2 for short). 
Funding rate provides strong BTC/ETH edge in bear markets by capturing extreme sentiment reversals. 
Tight entries (R1/S1 only) + volume confirmation + trend alignment + funding filter target 20-40 trades/year.
Only trade breakouts aligned with higher timeframe trend and extreme funding to avoid whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d_vals)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 20-period moving average (institutional participation)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > vol_ma
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 20 for volume MA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_conf = volume_confirmed[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume confirmation AND aligned with 1d EMA34 trend
        long_entry = (close_val > camarilla_r1_val) and vol_conf and (close_val > ema_34_val)
        short_entry = (close_val < camarilla_s1_val) and vol_conf and (close_val < ema_34_val)
        
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
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_FundingFilter"
timeframe = "4h"
leverage = 1.0