#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Volume-Weighted RSI (VWRSI) with 4h trend filter and 1d momentum confirmation
# VWRSI weights RSI gains/losses by volume to emphasize institutional participation
# Uses 4h EMA20 for trend direction and 1d ROC for momentum bias
# Volume confirmation filters low-conviction moves
# Target: 60-150 total trades over 4 years (15-37/year) with disciplined entries
name = "1h_VWRSI_4hEMA20_1dROC"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d ROC for momentum bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    roc_10_1d = (pd.Series(df_1d['close']).pct_change(periods=10).fillna(0)).values
    roc_10_1d_aligned = align_htf_to_ltf(prices, df_1d, roc_10_1d)
    
    # Volume-Weighted RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Weight gains and losses by volume
    vol_weighted_gain = gain * volume
    vol_weighted_loss = loss * volume
    
    # Smoothed averages with volume weighting
    avg_gain = pd.Series(vol_weighted_gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(vol_weighted_loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    vwrsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.2 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwrsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(roc_10_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VWRSI < 30 (oversold) + above 4h EMA20 + positive 1d ROC + volume confirmation
            if (vwrsi[i] < 30 and 
                close[i] > ema_20_4h_aligned[i] and 
                roc_10_1d_aligned[i] > 0 and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: VWRSI > 70 (overbought) + below 4h EMA20 + negative 1d ROC + volume confirmation
            elif (vwrsi[i] > 70 and 
                  close[i] < ema_20_4h_aligned[i] and 
                  roc_10_1d_aligned[i] < 0 and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if VWRSI > 50 (mean reversion) or breaks below 4h EMA20
            if (vwrsi[i] > 50) or (close[i] < ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if VWRSI < 50 (mean reversion) or breaks above 4h EMA20
            if (vwrsi[i] < 50) or (close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals