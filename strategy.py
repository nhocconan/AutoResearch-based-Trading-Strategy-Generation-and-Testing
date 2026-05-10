#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Combines Camarilla pivot levels on 12h with 1d EMA trend filter and volume confirmation.
# Uses Camarilla R1/S1 levels for breakout entries, trades only in direction of 1d EMA34 trend.
# Volume confirmation (>1.5x average) reduces false breaks. Designed for 12-37 trades/year to avoid fee drag.

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate 12-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12 = np.full(n, np.nan)
    for i in range(12, n):
        atr_12[i] = np.nanmean(tr[i-11:i+1])
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 12h (based on previous bar)
    camarilla_R1 = np.full(n, np.nan)
    camarilla_S1 = np.full(n, np.nan)
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            range_prev = high[i-1] - low[i-1]
            camarilla_R1[i] = close[i-1] + range_prev * 1.1 / 12
            camarilla_S1[i] = close[i-1] - range_prev * 1.1 / 12
    
    # Volume average (24 periods = 2 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.nanmean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_R1[i]) or np.isnan(camarilla_S1[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of 1d EMA34 trend
            if close[i] > ema_34_1d_aligned[i]:  # Uptrend
                # Long: Breakout above Camarilla R1 with volume confirmation
                if close[i] > camarilla_R1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below Camarilla S1 with volume confirmation
                if close[i] < camarilla_S1[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA34 or stoploss hit
            if close[i] < ema_34_1d_aligned[i] or (i > 0 and low[i] < high[i-1] - 2.0 * atr_12[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA34 or stoploss hit
            if close[i] > ema_34_1d_aligned[i] or (i > 0 and high[i] > low[i-1] + 2.0 * atr_12[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals