#!/usr/bin/env python3
"""
4h_RSI_40_60_TrendPullback_VolumeFilter
Long: RSI < 40 + price > EMA50 + volume spike + 1d trend up
Short: RSI > 60 + price < EMA50 + volume spike + 1d trend down
Exit: RSI crosses 50 or trend flips
Designed for 4h timeframe with moderate trade frequency and trend-following + mean reversion blend.
Works in bull via pullbacks to EMA50, works in bear via bounces off EMA50 in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA50 on 4h
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_50[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_ema50 = price > ema_50[i]
        below_ema50 = price < ema_50[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI < 40 (oversold), price above EMA50, uptrend, volume spike
            if (rsi_val < 40 and above_ema50 and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 60 (overbought), price below EMA50, downtrend, volume spike
            elif (rsi_val > 60 and below_ema50 and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: RSI crosses above 50 or trend flips down
            if rsi_val > 50 or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: RSI crosses below 50 or trend flips up
            if rsi_val < 50 or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_RSI_40_60_TrendPullback_VolumeFilter"
timeframe = "4h"
leverage = 1.0