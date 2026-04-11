#!/usr/bin/env python3
# 12h_1d_cci_rsi_volume_v1
# Strategy: 12h CCI combined with 1d RSI and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: CCI identifies cyclical turning points while RSI confirms overbought/oversold conditions. 
# Volume ensures conviction. Works in both bull and bear markets by fading extremes during high volume.
# Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_rsi_volume_v1"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # CCI(20) on 12h data
    typical_price = (high + low + close) / 3
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mean_dev = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    cci = (typical_price - tp_mean) / (0.015 * tp_mean_dev)
    cci_values = cci.values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(cci_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry logic: CCI extreme + RSI confirmation + volume
        if (cci_values[i] < -100 and rsi_1d_aligned[i] < 30 and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (cci_values[i] > 100 and rsi_1d_aligned[i] > 70 and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone
        elif position == 1 and cci_values[i] > -50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci_values[i] < 50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals