#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for ATR and volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility filter
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-period ATR for entry trigger
    tr6_1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr6_2 = np.abs(high[1:] - close[:-1])
    tr6_3 = np.abs(low[1:] - close[:-1])
    tr6 = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr6_1, np.maximum(tr6_2, tr6_3))])
    atr_6 = pd.Series(tr6).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # Need weekly EMA50 (50), daily ATR (14), volume MA20 (20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(atr_6[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volatility filter: current 6-period ATR > 1.5 * daily ATR
        vol_filter = atr_6[i] > (1.5 * atr_1d_aligned[i])
        
        # Volume filter: volume spike
        vol_spike = volume[i] > (2.0 * volume_ma20[i])
        
        if position == 0:
            # Long: uptrend + volatility expansion + volume spike
            if uptrend and vol_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volatility expansion + volume spike
            elif downtrend and vol_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility contraction
            if not uptrend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility contraction
            if not downtrend or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyEMA50Trend_VolExpansion_VolumeSpike"
timeframe = "6h"
leverage = 1.0