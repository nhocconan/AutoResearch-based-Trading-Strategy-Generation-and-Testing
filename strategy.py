#!/usr/bin/env python3

"""
Hypothesis: Daily KAMA trend with RSI mean-reversion and volume confirmation.
This strategy uses the 1-day Kaufman Adaptive Moving Average to determine trend direction
and RSI for mean-reversion entries, filtered by volume spikes. The adaptive nature of
KAMA helps it respond quickly in trending markets while avoiding whipsaws in ranging
markets. Works in both bull and bear regimes by following the higher-timeframe trend.
Target: 15-25 trades/year per symbol.
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
    
    # Load daily data - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA (10-period efficiency ratio)
    close_1d = pd.Series(df_1d['close'].values)
    # Efficiency ratio: |change| / sum(|changes|)
    change = close_1d.diff().abs()
    direction = (close_1d - close_1d.shift(10)).abs()
    er = direction / change.rolling(10).sum()
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * 0.288 + 0.064) ** 2  # fast=2/(2+2), slow=2/(30+1)
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = kama
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate daily RSI (14-period)
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Calculate daily volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_avg_20 = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price crosses above KAMA with oversold RSI and volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                volume[i] > 1.8 * vol_avg_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below KAMA with overbought RSI and volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  volume[i] > 1.8 * vol_avg_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to KAMA or RSI reverts
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below KAMA or RSI > 50
                if close[i] < kama_aligned[i] or rsi_aligned[i] > 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above KAMA or RSI < 50
                if close[i] > kama_aligned[i] or rsi_aligned[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_Volume_MeanRev"
timeframe = "1d"
leverage = 1.0