#!/usr/bin/env python3
# 4h_1d_cci_trend_volume_v1
# Strategy: 4h CCI with 1d trend and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. Combined with 1d trend (EMA50) and volume confirmation, it captures mean-reversion in trending markets. Works in both bull and bear by trading pullbacks in the direction of the 1d trend. Designed for low trade frequency to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_trend_volume_v1"
timeframe = "4h"
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
    
    # CCI on 4h (20-period)
    period = 20
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=period, min_periods=period).mean()
    mad = pd.Series(tp).rolling(window=period, min_periods=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    cci_values = cci.values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(cci_values[i]) or np.isnan(ema_50_1d_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 20-period average
        vol_confirm = vol_1d_aligned[i] > vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: CCI < -100 (oversold) AND uptrend AND volume confirmation
        if cci_values[i] < -100 and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI > 100 (overbought) AND downtrend AND volume confirmation
        elif cci_values[i] > 100 and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI crosses back above -100 (for long) or below 100 (for short)
        elif position == 1 and cci_values[i] > -100:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_values[i] < 100:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals