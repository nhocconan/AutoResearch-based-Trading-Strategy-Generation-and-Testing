#!/usr/bin/env python3
# 12h_1d_cci_volume_v1
# Strategy: 12h CCI with 1d EMA trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: CCI > 100 indicates strong bullish momentum, CCI < -100 indicates strong bearish momentum.
# Combined with 1d EMA trend filter to ensure alignment with higher timeframe trend and volume confirmation
# to filter weak signals. This strategy aims to capture strong trends while avoiding false signals in
# choppy markets, working in both bull and bear regimes by following institutional momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_cci_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_ma = pd.Series(typical_price).rolling(window=20, min_periods=20).mean()
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci = (typical_price - tp_ma.values) / (0.015 * tp_mad.values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_ratio = pd.Series(volume) / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(vol_ratio.iloc[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_ratio.iloc[i] > 1.5
        
        # Entry conditions
        # Long: CCI > 100 (strong bullish momentum) + price above 1d EMA50 (uptrend) + volume confirmation
        if vol_confirmed and cci[i] > 100 and close[i] > ema_50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI < -100 (strong bearish momentum) + price below 1d EMA50 (downtrend) + volume confirmation
        elif vol_confirmed and cci[i] < -100 and close[i] < ema_50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: loss of momentum or trend reversal
        elif position == 1 and (cci[i] <= 0 or close[i] < ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci[i] >= 0 or close[i] > ema_50_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals