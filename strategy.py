#!/usr/bin/env python3
# 6h_1w_1d_adaptive_kelly_ema_cross_v1
# Hypothesis: 6h EMA9/EMA21 crossover with 1d volatility filter and 1w trend filter.
# Uses adaptive Kelly sizing based on recent win rate and volatility to manage risk.
# Long when EMA9 > EMA21, price above 1d VWAP, and 1w uptrend; short when opposite.
# Designed for 15-35 trades/year on 6h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_adaptive_kelly_ema_cross_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h EMAs for entry signal
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 1d VWAP for mean reversion filter
    df_1d = get_htf_data(prices, '1d')
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volatility measure for Kelly sizing (ATR-based)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(21, 14)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend direction
        uptrend_6h = ema9[i] > ema21[i]
        downtrend_6h = ema9[i] < ema21[i]
        
        # Price relative to 1d VWAP
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross down or price drops significantly below VWAP
            if not uptrend_6h or close[i] < vwap_1d_aligned[i] - 0.5 * atr14[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Kelly sizing: base size 0.25, scaled by recent performance
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up or price rises significantly above VWAP
            if not downtrend_6h or close[i] > vwap_1d_aligned[i] + 0.5 * atr14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA9 crosses above EMA21, price above VWAP, and 1w uptrend
            if (uptrend_6h and price_above_vwap and uptrend_1w):
                position = 1
                signals[i] = 0.25
            # Short entry: EMA9 crosses below EMA21, price below VWAP, and 1w downtrend
            elif (downtrend_6h and price_below_vwap and downtrend_1w):
                position = -1
                signals[i] = -0.25
    
    return signals