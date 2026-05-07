#!/usr/bin/env python3
"""
6h_Relative_Strength_Index_with_Volume_and_Trend
Hypothesis: RSI(14) combined with volume confirmation and daily trend filter.
In bull markets (price > 200-day SMA), buy when RSI crosses above 30 with volume.
In bear markets (price < 200-day SMA), sell when RSI crosses below 70 with volume.
Uses daily timeframe for trend and 6-hour for entry timing to reduce noise and whipsaws.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RSI_Volume_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 200-day SMA for trend filter
    sma_200_1d = pd.Series(df_1d['close']).rolling(window=200, min_periods=200).mean().values
    sma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need 200 periods for SMA200 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(sma_200_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 200-day SMA
        bull_market = close[i] > sma_200_1d_aligned[i]
        bear_market = close[i] < sma_200_1d_aligned[i]
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: RSI crosses above 30 in bull market with volume
            long_entry = (rsi[i] > 30) and (rsi[i-1] <= 30) and bull_market and volume_confirm
            # Short: RSI crosses below 70 in bear market with volume
            short_entry = (rsi[i] < 70) and (rsi[i-1] >= 70) and bear_market and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses below 50 or trend changes to bear
            if (rsi[i] < 50) or (not bull_market):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses above 50 or trend changes to bull
            if (rsi[i] > 50) or (not bear_market):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals