#!/usr/bin/env python3
"""
4h Chaikin Money Flow + Trend Filter
Hypothesis: CMF measures buying/selling pressure. In trending markets, price follows CMF direction.
Long when CMF > 0 and price above 1d EMA50, short when CMF < 0 and price below 1d EMA50.
Exit when CMF crosses zero or stoploss hits. Works in bull (buy strength) and bear (sell weakness).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14320_4h_cmf_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA for daily trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (20)
    cmf_period = 20
    mf_multiplier = np.where((high - low) != 0, ((close - low) - (high - close)) / (high - low), 0)
    mf_volume = mf_multiplier * volume
    mf_volume_sum = pd.Series(mf_volume).rolling(window=cmf_period, min_periods=cmf_period).sum().values
    volume_sum = pd.Series(volume).rolling(window=cmf_period, min_periods=cmf_period).sum().values
    cmf = mf_volume_sum / (volume_sum + 1e-10)
    
    # Volume filter: avoid low volume periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (0.8 * vol_ma)  # Require at least 80% of average volume
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(cmf_period, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(cmf[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: CMF turns negative OR stoploss
            if cmf[i] <= 0 or close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: CMF turns positive OR stoploss
            if cmf[i] >= 0 or close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: CMF extreme + trend alignment + volume
            long_setup = (cmf[i] > 0.05) and (close[i] > ema_1d_aligned[i]) and vol_filter[i]
            short_setup = (cmf[i] < -0.05) and (close[i] < ema_1d_aligned[i]) and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals