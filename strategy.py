#!/usr/bin/env python3
# 6h_1d_ccitrade_v1
# Strategy: 6h CCI combined with 1d ADX for trend strength and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. In trending markets (ADX>25), we trade pullbacks in trend direction. In ranging markets (ADX<20), we mean-revert at extremes. Volume confirms momentum. Works in both bull/bear by adapting to trend regime.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ccitrade_v1"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], 0.0])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_ma = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_ma = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h CCI(20)
    tp = (high + low + close) / 3
    tp_ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - tp_ma) / (0.015 * tp_mad)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d[i]) or np.isnan(cci[i]) or 
            np.isnan(tp_mad[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Regime filters
        trending = adx_1d[i] > 25
        ranging = adx_1d[i] < 20
        
        # Entry logic
        if trending:
            # In trending markets: buy pullbacks in uptrend, sell rallies in downtrend
            if cci[i] < -100 and close[i] > close[i-1] and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif cci[i] > 100 and close[i] < close[i-1] and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
        elif ranging:
            # In ranging markets: mean reversion at extremes
            if cci[i] < -200 and vol_confirm[i] and position != 1:
                position = 1
                signals[i] = 0.25
            elif cci[i] > 200 and vol_confirm[i] and position != -1:
                position = -1
                signals[i] = -0.25
        # Exit: opposite signal or volatility expansion
        elif position == 1 and (cci[i] > 0 or (trending and cci[i] > 50)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci[i] < 0 or (trending and cci[i] < -50)):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals