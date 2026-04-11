#!/usr/bin/env python3
# 4h_1d_cci_rsi_volume_v2
# Strategy: 4h CCI (Commodity Channel Index) combined with RSI and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies cyclical overbought/oversold conditions. In ranging markets, 
# CCI > 100 with RSI > 70 indicates overextended long (short signal), CCI < -100 with 
# RSI < 30 indicates oversold short (long signal). Volume confirmation filters weak signals.
# Works in both bull and bear markets by fading extremes during consolidation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_rsi_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # CCI calculation: Typical price, SMA, Mean Deviation
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mean_dev = tp_series.rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - sma_tp.values) / (0.015 * mean_dev.values)
    
    # RSI calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(rsi.iloc[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion logic: fade extremes with volume confirmation
        if (cci[i] > 100 and rsi.iloc[i] > 70 and vol_confirm[i] and position != -1):
            # Overbought: short signal
            position = -1
            signals[i] = -0.25
        elif (cci[i] < -100 and rsi.iloc[i] < 30 and vol_confirm[i] and position != 1):
            # Oversold: long signal
            position = 1
            signals[i] = 0.25
        # Exit when CCI returns to neutral zone
        elif position == 1 and cci[i] >= -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] <= 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals