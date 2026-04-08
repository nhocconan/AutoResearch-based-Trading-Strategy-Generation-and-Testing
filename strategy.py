#!/usr/bin/env python3
# 1d_1w_vwap_momentum_trend_v1
# Hypothesis: Daily trend following using weekly VWAP as dynamic support/resistance and momentum confirmation.
# Long: price > weekly VWAP AND daily momentum > 0
# Short: price < weekly VWAP AND daily momentum < 0
# Exit: price crosses weekly VWAP or momentum reverses.
# Designed to capture multi-day trends while avoiding whipsaws in ranging markets.
# Weekly VWAP provides institutional reference point; momentum filters noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_vwap_momentum_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily momentum (5-period ROC)
    momentum = np.full(n, np.nan)
    for i in range(5, n):
        momentum[i] = (close[i] - close[i-5]) / close[i-5]
    
    # Weekly VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    vwap_1w = np.full(len(typical_price_1w), np.nan)
    cum_vol_price = 0.0
    cum_vol = 0.0
    
    for i in range(len(typical_price_1w)):
        pv = typical_price_1w[i] * df_1w['volume'].values[i]
        vol = df_1w['volume'].values[i]
        cum_vol_price += pv
        cum_vol += vol
        if cum_vol > 0:
            vwap_1w[i] = cum_vol_price / cum_vol
    
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        mom = momentum[i]
        vwap = vwap_1w_aligned[i]
        price = close[i]
        
        if np.isnan(mom) or np.isnan(vwap):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if mom <= 0 or price < vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if mom >= 0 or price > vwap:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if mom > 0 and price > vwap:
                position = 1
                signals[i] = 0.25
            elif mom < 0 and price < vwap:
                position = -1
                signals[i] = -0.25
    
    return signals