#!/usr/bin/env python3
# 1d_1w_volatility_breakout_v1
# Hypothesis: Breakouts of weekly volatility channels on 1d timeframe with volatility filter.
# Long when price breaks above weekly ATR-based upper band in low volatility regime.
# Short when price breaks below weekly ATR-based lower band in low volatility regime.
# Uses weekly ATR(14) to set channels and daily ATR(14) to filter for low volatility regimes.
# Designed for low trade frequency (7-25/year) to avoid fee drag, works in bull/bear via volatility regime filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for volatility channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly ATR(14) for channel width
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Weekly channels: midpoint ± ATR multiplier
    midpoint_1w = (high_1w + low_1w) / 2
    upper_band_1w = midpoint_1w + 1.5 * atr14_1w
    lower_band_1w = midpoint_1w - 1.5 * atr14_1w
    
    # Align weekly channels to daily
    upper_band_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_band_1w)
    lower_band_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_band_1w)
    
    # Daily ATR(14) for volatility regime filter
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr14_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: low volatility when ATR < 20-period SMA of ATR
    atr_ma = pd.Series(atr14_d).rolling(window=20, min_periods=20).mean().values
    low_volatility = atr14_d < atr_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(upper_band_1w_aligned[i]) or np.isnan(lower_band_1w_aligned[i]) or np.isnan(low_volatility[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below lower band or volatility increase
            if close[i] < lower_band_1w_aligned[i] or not low_volatility[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above upper band or volatility increase
            if close[i] > upper_band_1w_aligned[i] or not low_volatility[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_volatility[i]:
                # Long entry: price above upper band
                if close[i] > upper_band_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below lower band
                elif close[i] < lower_band_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals