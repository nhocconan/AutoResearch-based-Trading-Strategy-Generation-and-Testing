#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 1-day ATR volatility regime and daily EMA34 trend filter.
# Uses 1-day ATR ratio to filter volatility regimes: only trade when ATR(7)/ATR(30) > 1.3 (expanding volatility).
# Combined with daily EMA34 for trend direction to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-38/year) with size 0.25.

name = "4h_ATR_Volatility_Regime_EMA34_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day ATR for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR(7) and ATR(30)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with same length
    
    # ATR(7) and ATR(30)
    atr_7 = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Volatility regime: ATR(7)/ATR(30) > 1.3 indicates expanding volatility
    vol_regime = atr_7 / atr_30 > 1.3
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Volume filter: current volume > 1.8x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: expanding volatility + 1d uptrend + volume spike
            if vol_regime_aligned[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: expanding volatility + 1d downtrend + volume spike
            elif vol_regime_aligned[i] and trend_down[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility contraction or trend reversal
            if not vol_regime_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility contraction or trend reversal
            if not vol_regime_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals