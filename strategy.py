#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts with 12h trend filter and volume confirmation work in both bull and bear markets.
In bull markets: price breaks above R1 (resistance) with 12h uptrend → long.
In bear markets: price breaks below S1 (support) with 12h downtrend → short.
Uses discrete sizing (0.30) and ATR-based stoploss to limit drawdown. Target: 75-150 trades over 4 years.
Camarilla levels provide intraday structure that adapts to volatility, and 12h trend filters out counter-trend noise.
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
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 12h data for HTF trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We only need R1 and S1: R1 = close + 1.125*(high-low), S1 = close - 1.125*(high-low)
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Previous 12h bar's range
    range_12h = h_12h - l_12h
    camarilla_r1 = c_12h + 1.125 * range_12h
    camarilla_s1 = c_12h - 1.125 * range_12h
    
    # Align Camarilla levels to 4h timeframe (they represent the completed 12h bar)
    r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]]) if len(tr1) > 0 else [0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    bars_since_entry = 0
    
    # Start after warmup (need 50 for EMA, 14 for ATR)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        atr_val = atr[i]
        
        # Long logic: price breaks above R1 with volume spike and 12h uptrend
        long_condition = (close_val > r1_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S1 with volume spike and 12h downtrend
        short_condition = (close_val < s1_val) and volume_spike[i] and (close_val < ema_val)
        
        # Stoploss logic: ATR-based
        stop_long = position == 1 and close_val < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and close_val > (entry_price + 2.0 * atr_val)
        
        # Minimum holding period: 4 bars (to reduce churn)
        if position != 0 and bars_since_entry < 4:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
            entry_price = close_val  # Approximate entry price for stoploss
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
            entry_price = close_val  # Approximate entry price for stoploss
        elif position == 1 and (close_val < ema_val or stop_long):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and (close_val > ema_val or stop_short):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0