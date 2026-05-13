#!/usr/bin/env python3
name = "1d_RM_Acme_1W_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA(34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily ATR(14) for volatility
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1]) if i >= 14 else np.nan
    
    # Daily Donchian channel (20)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
            vol_filter = volume[i] > 1.5 * vol_ma_20
        else:
            vol_filter = False
        
        if position == 0:
            # LONG: price breaks above Donchian high + weekly uptrend + volume
            if close[i] > donch_high[i] and close[i] > ema34_1w_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low + weekly downtrend + volume
            elif close[i] < donch_low[i] and close[i] < ema34_1w_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian low or trend breaks
            if close[i] < donch_low[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian high or trend breaks
            if close[i] > donch_high[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals