#!/usr/bin/env python3
# 4h_1d_cci_volume_v1
# Strategy: 4h CCI(20) extremes with 1-day VWAP filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: CCI identifies overbought/oversold conditions. 1-day VWAP acts as dynamic support/resistance.
# Volume > 1.5x 20-period average confirms institutional participation. Works in both bull and bear markets
# by fading extremes during ranging periods and following trends during breakouts. Low trade frequency
# (~15-30/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h CCI(20)
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - ma_tp) / (0.015 * mad)
    
    # 1-day VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # 4h volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # CCI signals: >100 overbought, <-100 oversold
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Price relative to 1-day VWAP
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Entry conditions
        # Long: CCI oversold AND price above VWAP AND volume confirmation
        if cci_oversold and price_above_vwap and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: CCI overbought AND price below VWAP AND volume confirmation
        elif cci_overbought and price_below_vwap and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: CCI returns to neutral zone (-50 to 50)
        elif position == 1 and cci[i] < 50:
            position = 0
            signals[i] = 0.0
        elif position == -1 and cci[i] > -50:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals