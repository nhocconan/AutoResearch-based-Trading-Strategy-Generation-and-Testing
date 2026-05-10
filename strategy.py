#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) on 4h timeframe for breakout entries.
# Enters long when price breaks above R1 with volume spike and 1d uptrend (close > EMA34).
# Enters short when price breaks below S1 with volume spike and 1d downtrend (close < EMA34).
# Exits when price returns to the 1d VWAP or when volatility collapses (low volume).
# Designed for moderate trade frequency (target: 25-40 trades/year) with strong trend follow-through.
# Works in bull markets via breakouts and in bear markets via breakdowns with trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 1d VWAP for exit condition
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # 1d EMA(34) for trend filter
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1_1d = close_1d + camarilla_range
    s1_1d = close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume spike detection: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vwap_1d[i]) if i < len(vwap_1d) else True or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Need to check if we have valid VWAP value for current day
        if i >= len(vwap_1d):
            vwap_current = vwap_1d[-1] if len(vwap_1d) > 0 else 0
        else:
            vwap_current = vwap_1d[i]
        
        if position == 0:
            # Long: Break above R1 + volume spike + 1d uptrend (close > EMA34)
            if close[i] > r1_1d_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 + volume spike + 1d downtrend (close < EMA34)
            elif close[i] < s1_1d_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns to 1d VWAP or volume dries up (no spike)
            if close[i] <= vwap_current or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns to 1d VWAP or volume dries up (no spike)
            if close[i] >= vwap_current or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals