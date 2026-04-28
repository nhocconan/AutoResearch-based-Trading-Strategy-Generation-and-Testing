#!/usr/bin/env python3
"""
1d_WoodieCCI_Trend_Reversal_1wTrend
Hypothesis: Woodie CCI (Commodity Channel Index) on daily timeframe identifies overbought/oversold conditions in trending markets. 
Combined with 1-week EMA trend filter and volume confirmation to catch reversals in both bull and bear markets. 
Target: 15-25 trades/year with strict entry conditions to minimize fee drag.
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Woodie CCI on daily data (using daily high/low/close)
    # First get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price for Woodie CCI
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # 14-period CCI (Woodie uses 14, not 20)
    cci_period = 14
    ma_tp = pd.Series(tp_1d).rolling(window=cci_period, min_periods=cci_period).mean()
    md_tp = pd.Series(tp_1d).rolling(window=cci_period, min_periods=cci_period).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    # Woodie CCI formula: (TP - SMA) / (0.015 * Mean Deviation)
    cci_1d = (tp_1d - ma_tp.values) / (0.015 * md_tp.values)
    
    # Align CCI to 1d timeframe (already daily, but need to align to lower timeframe)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d.values)
    
    # Volume confirmation: >1.3x 20-period MA on daily volume
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(cci_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Woodie CCI signals: >100 overbought, <-100 oversold
        cci_overbought = cci_1d_aligned[i] > 100
        cci_oversold = cci_1d_aligned[i] < -100
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.3 * vol_ma_20_1d_aligned[i])
        
        # Entry logic: reversal from extreme CCI levels with volume, counter to weekly trend
        # In uptrend, look for oversold (buy dip); in downtrend, look for overbought (sell rally)
        long_entry = vol_confirm and cci_oversold and uptrend  # Buy dip in uptrend
        short_entry = vol_confirm and cci_overbought and downtrend  # Sell rally in downtrend
        
        # Exit logic: CCI returns to neutral zone or trend reversal
        long_exit = (cci_1d_aligned[i] >= 0) or (not uptrend)
        short_exit = (cci_1d_aligned[i] <= 0) or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WoodieCCI_Trend_Reversal_1wTrend"
timeframe = "1d"
leverage = 1.0