#!/usr/bin/env python3
# 4h_VWAP_Reversion_Pullback
# Hypothesis: 4h mean reversion to VWAP with trend filter and volume confirmation.
# Strategy buys when price pulls back to VWAP in uptrend with volume confirmation,
# and sells when price rallies to VWAP in downtrend with volume confirmation.
# Works in both bull and bear markets by trading pullbacks to VWAP in the direction
# of the higher timeframe trend. Uses VWAP as dynamic support/resistance.

timeframe = "4h"
name = "4h_VWAP_Reversion_Pullback"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate VWAP for 4h chart (cumulative from session start)
    typical_price = (high + low + close) / 3.0
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to VWAP from below in uptrend with volume
            if (low[i] <= vwap[i] <= high[i] and  # price touches VWAP
                close[i] > vwap[i] and           # closes above VWAP (bounce)
                close[i] > ema_50_1d_aligned[i] and  # uptrend filter
                volume[i] > 1.5 * vol_ma[i]):    # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: price rallies to VWAP from above in downtrend with volume
            elif (low[i] <= vwap[i] <= high[i] and  # price touches VWAP
                  close[i] < vwap[i] and           # closes below VWAP (rejection)
                  close[i] < ema_50_1d_aligned[i] and  # downtrend filter
                  volume[i] > 1.5 * vol_ma[i]):    # volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below VWAP (break of support)
            if close[i] < vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above VWAP (break of resistance)
            if close[i] > vwap[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals