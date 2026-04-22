# 12h_1wATRTrendFilter_1dBreakout_v1
# Hypothesis: Weekly ATR trend + daily breakout on 12h timeframe
# Weekly ATR > 50-period mean indicates strong trend regime.
# Daily breakout above/below ATR channel provides directional signal.
# Uses 12h timeframe to limit trade frequency (12-37 trades/year target).
# Works in bull/bear via trend filter - only trades in strong weekly volatility regimes.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Weekly ATR for trend regime filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for weekly ATR
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    
    # Daily ATR for breakout channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1d = high_1d[1:] - low_1d[1:]
    tr2d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1d, np.maximum(tr2d, tr3d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Align indicators to 12h timeframe
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_ma_1w_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr_1w_val = atr_1w_aligned[i]
        atr_ma_1w_val = atr_ma_1w_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Trend regime: weekly ATR above its mean (strong volatility/trend)
        trend_regime = atr_1w_val > atr_ma_1w_val
        
        if position == 0 and trend_regime:
            # Long: price breaks above close + 1.0 * daily ATR
            if price > close[i-1] + 1.0 * atr_1d_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below close - 1.0 * daily ATR
            elif price < close[i-1] - 1.0 * atr_1d_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit: mean reversion to previous close or trend regime ends
            mean_rev = (position == 1 and price < close[i-1]) or (position == -1 and price > close[i-1])
            trend_end = not trend_regime
            
            if mean_rev or trend_end:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_1wATRTrendFilter_1dBreakout_v1"
timeframe = "12h"
leverage = 1.0