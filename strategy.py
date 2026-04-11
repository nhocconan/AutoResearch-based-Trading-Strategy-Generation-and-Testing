#!/usr/bin/env python3
"""
6h_1d_cci_extreme_volume_v1
Strategy: 6h CCI extreme levels with volume confirmation and 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses 6h Commodity Channel Index (CCI) to identify overbought/oversold conditions (>100 or <-100). Extreme CCI readings often precede reversals. Combined with volume spikes (>2x average) for confirmation and filtered by 1d EMA50 trend direction to trade with higher timeframe momentum. Works in both bull and bear markets by taking mean-reversion trades at extremes while respecting the dominant trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_extreme_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 6h CCI(20)
    tp = (high + low + close) / 3.0  # Typical Price
    tp_ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci = (tp - tp_ma) / (0.015 * tp_mad)
    cci = np.where(tp_mad == 0, 0, cci)  # Avoid division by zero
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # CCI extreme conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: CCI oversold with volume in uptrend (mean reversion up)
        long_signal = cci_oversold and vol_confirmed and uptrend_1d
        
        # Short: CCI overbought with volume in downtrend (mean reversion down)
        short_signal = cci_overbought and vol_confirmed and downtrend_1d
        
        # Exit when CCI returns to neutral zone (-50 to 50)
        exit_long = position == 1 and cci[i] > -50
        exit_short = position == -1 and cci[i] < 50
        
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