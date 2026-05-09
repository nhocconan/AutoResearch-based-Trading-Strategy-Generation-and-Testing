#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    # Camarilla pivot levels: H3, L3, H4, L4
    pivot = (high + low + close) / 3
    range_val = high - low
    h3 = close + range_val * 1.1 / 2
    l3 = close - range_val * 1.1 / 2
    h4 = close + range_val * 1.1
    l4 = close - range_val * 1.1
    return h3, l3, h4, l4

def calculate_atr(high, low, close, period):
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla, trend, and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    h3_1d, l3_1d, h4_1d, l4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d ATR for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 10)
    
    # Calculate 1d EMA34 trend
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: current 12h ATR > 0.8 * 1d ATR (ensures sufficient volatility)
    tr_12h = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = 0
    atr_12h = pd.Series(tr_12h).ewm(span=10, adjust=False, min_periods=10).mean().values
    vol_filter = atr_12h > (atr_1d_aligned * 0.8)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or
            np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        h3_val = h3_1d_aligned[i]
        l3_val = l3_1d_aligned[i]
        h4_val = h4_1d_aligned[i]
        l4_val = l4_1d_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        vol_filt = volume_filter[i]
        vol_filt2 = vol_filter[i]
        
        if position == 0:
            # Enter long: price > H3 + above EMA34 + volume filter + volatility filter
            if close[i] > h3_val and close[i] > ema34_val and vol_filt and vol_filt2:
                signals[i] = 0.25
                position = 1
            # Enter short: price < L3 + below EMA34 + volume filter + volatility filter
            elif close[i] < l3_val and close[i] < ema34_val and vol_filt and vol_filt2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < L3 or below EMA34
            if close[i] < l3_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > H3 or above EMA34
            if close[i] > h3_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals