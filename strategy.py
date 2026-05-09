#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_HybridBreakout_Trend_Scalp_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high and low for breakout levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 4h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Trend filter: 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volatility filter: ATR(7) > ATR(14) * 0.8 (avoid low volatility chop)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vol_filter = atr7 > (atr14 * 0.8)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 34)  # Need enough data for ATR14 and EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ph = prev_high_aligned[i]
        pl = prev_low_aligned[i]
        trend = ema34_1d_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Enter long: break above previous day's high with volume and above trend
            if close[i] > ph and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below previous day's low with volume and below trend
            elif close[i] < pl and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below previous day's low (mean reversion)
            if close[i] < pl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above previous day's high (mean reversion)
            if close[i] > ph:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals