#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4h trend filter (price above/below 4h EMA34) and volume spike filter (ATR ratio > 1.5). Camarilla pivots provide precise intraday support/resistance levels. 4h trend ensures alignment with higher timeframe momentum to avoid counter-trend trades. Volume spike confirms institutional participation. Discrete sizing 0.20 limits trades to target 15-37/year. Works in bull/bear via 4h trend filter.
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
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    open_time = prices['open_time']
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate ATR(14) for volume regime
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate Camarilla pivots for each 1h bar using previous day's OHLC
    # We need to get the previous day's high, low, close for each 1h bar
    # Approach: resample to 1d to get daily OHLC, then shift by 1 to get previous day
    # But we cannot resample - instead we'll use the 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get previous day's OHLC for each 1h bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align 1d data to 1h timeframe (previous day's values)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    range_1d = high_1d_aligned - low_1d_aligned
    r1 = close_1d_aligned + range_1d * 1.1 / 12
    s1 = close_1d_aligned - range_1d * 1.1 / 12
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (34 for EMA, 50 for ATR ratio)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session.iloc[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(high_1d_aligned[i]) or np.isnan(low_1d_aligned[i]) or
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1[i]
        s1_val = s1[i]
        ema_34_val = ema_34_4h_aligned[i]
        vol_spike = atr_ratio[i] > 1.5  # volume regime
        size = fixed_size
        
        # Entry conditions: Camarilla R1/S1 breakout with volume spike AND aligned with 4h EMA34 trend
        # Long: price breaks above R1 (bullish breakout)
        # Short: price breaks below S1 (bearish breakout)
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
            # Long - exit when price re-enters Camarilla range or trend reversal
            if close_val < r1_val and close_val > s1_val:  # back inside Camarilla range
                signals[i] = 0.0
                position = 0
            elif close_val < ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters Camarilla range or trend reversal
            if close_val > s1_val and close_val < r1_val:  # back inside Camarilla range
                signals[i] = 0.0
                position = 0
            elif close_val > ema_34_val:  # trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0