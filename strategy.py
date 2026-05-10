#!/usr/bin/env python3
# 4h_1D_EMA_Cross_With_4H_CCI_Trend
# Hypothesis: Daily EMA34 establishes trend direction, 4h CCI(20) confirms momentum with
# overbought/oversold conditions, and 4h volume filter eliminates false signals.
# Works in bull markets by buying dips in uptrends, in bear markets by selling rallies
# in downtrends. Uses only 3 conditions to minimize trade frequency and fee drag.

name = "4h_1D_EMA_Cross_With_4H_CCI_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for CCI calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate CCI(20) on 4h data
    tp_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    sma_tp = tp_4h.rolling(window=20, min_periods=20).mean()
    mad = tp_4h.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci_4h = (tp_4h - sma_tp) / (0.015 * mad)
    cci_4h = cci_4h.replace([np.inf, -np.inf], np.nan).fillna(0).values
    cci_4h_aligned = align_htf_to_ltf(prices, df_4h, cci_4h)
    
    # Volume confirmation on 4h (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA34 (34), 4h CCI (20), volume MA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(cci_4h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # CCI conditions
        cci_oversold = cci_4h_aligned[i] < -100
        cci_overbought = cci_4h_aligned[i] > 100
        
        if position == 0:
            # Long entry: uptrend + CCI oversold + volume
            if uptrend and cci_oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + CCI overbought + volume
            elif downtrend and cci_overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or CCI becomes overbought
            if not uptrend or cci_4h_aligned[i] > 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or CCI becomes oversold
            if not downtrend or cci_4h_aligned[i] < -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals