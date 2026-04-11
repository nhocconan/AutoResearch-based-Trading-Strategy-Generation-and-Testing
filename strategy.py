#!/usr/bin/env python3
"""
12h_1d_1w_vwap_trend_reversion_v1
Strategy: 12h VWAP reversion with 1d/1w trend filter and volume confirmation
Timeframe: 12h
Leverage: 1.0
Hypothesis: Price tends to revert to VWAP in ranging markets but trends away in strong trends. 
Goes long when price deviates below VWAP in a 1d/1w uptrend with volume confirmation, 
short when price deviates above VWAP in a 1d/1w downtrend with volume confirmation. 
Uses 1d and 1w VWAP deviation z-score to avoid chop and capture mean reversion in trends.
Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 12-37 trades per year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_vwap_trend_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_numerator = (typical_price * volume).cumsum()
    vwap_denominator = volume.cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # VWAP deviation (normalized by ATR for volatility scaling)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr_14 == 0, 0.001, atr_14)
    vwap_dev = (close - vwap) / atr_safe
    
    # VWAP deviation z-score (20-period)
    vwap_dev_mean = pd.Series(vwap_dev).rolling(window=20, min_periods=20).mean().values
    vwap_dev_std = pd.Series(vwap_dev).rolling(window=20, min_periods=20).std().values
    vwap_dev_std_safe = np.where(vwap_dev_std == 0, 1, vwap_dev_std)
    vwap_zscore = (vwap_dev - vwap_dev_mean) / vwap_dev_std_safe
    
    # Volume confirmation (volume > 1.5x 20-period average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d VWAP for trend
    tp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_num_1d = (tp_1d * df_1d['volume']).cumsum()
    vwap_den_1d = df_1d['volume'].cumsum()
    vwap_1d = vwap_num_1d / vwap_den_1d
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d.values)
    
    # 1w VWAP for trend
    tp_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_num_1w = (tp_1w * df_1w['volume']).cumsum()
    vwap_den_1w = df_1w['volume'].cumsum()
    vwap_1w = vwap_num_1w / vwap_den_1w
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w.values)
    
    # Trend filter: price above/both VWAPs = uptrend, below both = downtrend
    price_close = close
    above_1d = price_close > vwap_1d_aligned
    above_1w = price_close > vwap_1w_aligned
    below_1d = price_close < vwap_1d_aligned
    below_1w = price_close < vwap_1w_aligned
    
    uptrend = above_1d & above_1w
    downtrend = below_1d & below_1w
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_zscore[i]) or np.isnan(vwap_dev[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals with volume confirmation
        # Long when price significantly below VWAP in uptrend with volume
        long_signal = (vwap_zscore[i] < -1.5) and vol_spike[i] and uptrend[i]
        
        # Short when price significantly above VWAP in downtrend with volume
        short_signal = (vwap_zscore[i] > 1.5) and vol_spike[i] and downtrend[i]
        
        # Exit when price returns near VWAP (z-score between -0.5 and 0.5)
        exit_long = position == 1 and (-0.5 <= vwap_zscore[i] <= 0.5)
        exit_short = position == -1 and (-0.5 <= vwap_zscore[i] <= 0.5)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals