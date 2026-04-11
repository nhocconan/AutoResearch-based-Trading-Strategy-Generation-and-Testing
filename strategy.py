#!/usr/bin/env python3
# 4h_1d_cci_rsi_volume_v1
# Strategy: 4h CCI + RSI + Volume Confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions (>100 or <-100). RSI confirms momentum (>50 for long, <50 for short). Volume ensures conviction. Works in both bull and bear markets by fading extremes with volume confirmation. Low frequency to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d CCI(20) for regime filter
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    sma_tp_20 = typical_price_1d.rolling(window=20, min_periods=20).mean()
    mad = typical_price_1d.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (typical_price_1d - sma_tp_20) / (0.015 * mad)
    cci_1d = cci_1d.values
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # 4h CCI(20) for signal
    typical_price = (high + low + close) / 3
    sma_tp_20 = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp_20) / (0.015 * mad)
    cci = cci.values
    
    # 4h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: current volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.3 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions: CCI extreme + RSI confirmation + volume
        if (cci[i] < -100 and rsi[i] < 50 and vol_confirm[i] and 
            cci_1d_aligned[i] < 0 and position != 1):  # Oversold + bearish 1d CCI
            position = 1
            signals[i] = 0.25
        elif (cci[i] > 100 and rsi[i] > 50 and vol_confirm[i] and 
              cci_1d_aligned[i] > 0 and position != -1):  # Overbought + bullish 1d CCI
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral or regime change
        elif position == 1 and (cci[i] > -50 or cci_1d_aligned[i] > 0):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci[i] < 50 or cci_1d_aligned[i] < 0):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals