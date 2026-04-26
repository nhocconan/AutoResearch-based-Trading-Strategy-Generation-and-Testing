#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_ATRStop_v1
Hypothesis: Camarilla pivot breakout with 1w trend filter and ATR-based stoploss. 
Long when price breaks above R1 with volume confirmation and 1w close > EMA(34). 
Short when price breaks below S1 with volume confirmation and 1w close < EMA(34).
Uses discrete sizing (0.25) to limit fee churn and ATR stoploss for risk control.
Target: 50-150 trades over 4 years = 12-37/year. Works in bull (trend continuation) and bear (counter-trend retracements) via 1w trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: R1, S1
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    R1 = close_1d + camarilla_range
    S1 = close_1d - camarilla_range
    
    # Align 1d Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[:1] = 0  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1w, ATR(14), volume MA(20)
    start_idx = max(34, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        regime_long = close_1w[i] > ema_34_1w_aligned[i]  # 1w uptrend
        regime_short = close_1w[i] < ema_34_1w_aligned[i]  # 1w downtrend
        
        if position == 0:
            # Long: price breaks above R1 with volume confirm AND 1w uptrend
            long_signal = (close_val > R1_aligned[i]) and vol_conf and regime_long
            
            # Short: price breaks below S1 with volume confirm AND 1w downtrend
            short_signal = (close_val < S1_aligned[i]) and vol_conf and regime_short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: ATR-based stoploss or trend flip
            if (close_val <= entry_price - 2.5 * atr[i]) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: ATR-based stoploss or trend flip
            if (close_val >= entry_price + 2.5 * atr[i]) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_ATRStop_v1"
timeframe = "12h"
leverage = 1.0