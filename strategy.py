#!/usr/bin/env python3
# 6h_1d_1w_cci_volume_breakout_v1
# Strategy: 6-hour CCI breakout with volume confirmation and 1d/1w trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: CCI(20) identifies overbought/oversold conditions; breakouts beyond ±100
# with volume confirmation (RVOL > 1.5) capture momentum. Trend filters from 1d EMA(50)
# and 1w EMA(20) ensure trades align with higher timeframe direction, reducing false
# signals in choppy markets. Works in bull by catching continuation breakouts and in
# bear by capturing breakdowns with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_cci_volume_breakout_v1"
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
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 6h CCI(20): (Typical Price - SMA(TP,20)) / (0.015 * Mean Deviation)
    tp = (high + low + close) / 3.0
    tp_series = pd.Series(tp)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean().values
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
    
    # 6h Relative Volume (RVOL): current volume / 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    rvol = volume / (vol_avg_20 + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(rvol[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # CCI breakout conditions
        bull_breakout = cci[i] > 100.0  # Break above overbought
        bear_breakout = cci[i] < -100.0  # Break below oversold
        
        # Volume confirmation: RVOL > 1.5
        vol_confirm = rvol[i] > 1.5
        
        # Trend filters: price above/below daily EMA50 and weekly EMA20
        uptrend = close[i] > ema_50_1d_aligned[i] and close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i] and close[i] < ema_20_1w_aligned[i]
        
        # Entry logic: CCI breakout + volume + trend alignment
        if bull_breakout and vol_confirm and uptrend and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_breakout and vol_confirm and downtrend and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite CCI breakout with volume confirmation
        elif position == 1 and bear_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_breakout and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals