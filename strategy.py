#!/usr/bin/env python3
"""
6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike
Hypothesis: RSI(2) extreme mean reversion on 6h timeframe, filtered by 1d EMA50 trend and volume spike (>2x average).
Long when RSI(2) < 10 and price > 1d EMA50 and volume > 2x average.
Short when RSI(2) > 90 and price < 1d EMA50 and volume > 2x average.
Exit on RSI(2) crossing back to neutral (40-60 range) or opposite extreme.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed for low trade frequency (12-37/year) to avoid fee drag while capturing short-term reversals in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(2) on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=2, adjust=False, min_periods=2).mean().values
    loss_ma = pd.Series(loss).ewm(span=2, adjust=False, min_periods=2).mean().values
    rs = np.divide(gain_ma, loss_ma, out=np.zeros_like(gain_ma), where=loss_ma!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA (50), RSI (2), volume MA (20)
    start_idx = max(50, 2, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: RSI(2) < 10, uptrend, volume spike
            long_signal = (rsi_val < 10) and (close_val > ema_50_1d_val) and (volume_val > 2.0 * vol_ma_val)
            # Short: RSI(2) > 90, downtrend, volume spike
            short_signal = (rsi_val > 90) and (close_val < ema_50_1d_val) and (volume_val > 2.0 * vol_ma_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: RSI(2) crosses back above 40 (mean reversion complete) or opposite extreme
            if (rsi_val > 40) or (rsi_val > 90):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: RSI(2) crosses back below 60 (mean reversion complete) or opposite extreme
            if (rsi_val < 60) or (rsi_val < 10):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI2_MeanReversion_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0