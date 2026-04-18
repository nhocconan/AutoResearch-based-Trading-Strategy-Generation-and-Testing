#!/usr/bin/env python3
"""
6h_SwingReversal_1dTrend_VolumeFilter
Hypothesis: On 6h timeframe, swing reversals identified by price crossing 6-period RSI(14) extremes (RSI<30 for long, RSI>70 for short) combined with 1d EMA(50) trend filter and volume spike (2.0x 20-period average) capture mean-reversion in trending markets. Works in both bull and bear: in bull markets, buys dips in uptrend; in bear markets, sells rallies in downtrend. Target: 20-40 trades/year.
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
    
    # Get daily data for EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close']
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6-period RSI(14) on 6h data
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) with volume spike and price above daily EMA (uptrend)
            if rsi_val < 30 and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) with volume spike and price below daily EMA (downtrend)
            elif rsi_val > 70 and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI returns to neutral (50) or breaks below daily EMA
            if rsi_val >= 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI returns to neutral (50) or breaks above daily EMA
            if rsi_val <= 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_SwingReversal_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0