#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h EMA Pullback with 4h Trend and 1d Filter
# Hypothesis: In strong 4h/1d trends, 1h EMA(21) pullbacks offer high-probability entries.
# Uses 4h EMA(50) for trend direction and 1d close > open as market regime filter.
# Target: 15-30 trades/year (60-120 total) to minimize fee drag.
# Works in bull/bear: trend filter prevents counter-trend trades.

name = "1h_ema_pullback_4h1d_trend_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    
    # Align 4h EMA to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d bull regime: close > open
    bull_regime_1d = (df_1d['close'].values > df_1d['open'].values).astype(float)
    bull_regime_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_regime_1d)
    
    # EMA(21) on 1h close
    ema_21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(bull_regime_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below EMA(21) or trend change
            if close[i] < ema_21[i] or close[i] < ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price above EMA(21) or trend change
            if close[i] > ema_21[i] or close[i] > ema_50_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Only trade in direction of 4h trend and 1d bull regime
            if bull_regime_1d_aligned[i] > 0.5:  # 1d bull regime
                if close[i] > ema_50_4h_aligned[i]:  # Uptrend
                    if close[i] <= ema_21[i]:  # Pullback to EMA
                        position = 1
                        signals[i] = 0.20
            else:  # 1d bear regime
                if close[i] < ema_50_4h_aligned[i]:  # Downtrend
                    if close[i] >= ema_21[i]:  # Pullback to EMA
                        position = -1
                        signals[i] = -0.20
    
    return signals