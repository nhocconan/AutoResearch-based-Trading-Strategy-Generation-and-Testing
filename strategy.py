#!/usr/bin/env python3
"""
1d_cci_momentum_1w_trend_v1
Hypothesis: On daily timeframe, use CCI (Commodity Channel Index) to detect momentum extremes combined with weekly trend filter. Enter long when CCI crosses above -100 in uptrend, short when CCI crosses below +100 in downtrend. Filter by weekly EMA trend and volatility contraction to avoid whipsaws. Designed to capture trend continuation moves while avoiding counter-trend noise. Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_cci_momentum_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate CCI on daily timeframe
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (typical_price - sma_tp) / (0.015 * mad)
    
    # Calculate weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volatility filter: low ATR ratio (contraction)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    # Avoid division by zero
    atr_ma = np.where(atr_ma == 0, 1e-10, atr_ma)
    atr_ratio = atr / atr_ma
    low_volatility = atr_ratio < 0.8  # Volatility contraction
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(ema_1w_aligned[i]) or 
            np.isnan(low_volatility[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (momentum fading) or volatility expands
            if cci[i] < 100 or not low_volatility[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (momentum fading) or volatility expands
            if cci[i] > -100 or not low_volatility[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in low volatility environments to avoid whipsaws
            if low_volatility[i]:
                # Long: CCI crosses above -100 in uptrend
                if cci[i] > -100 and cci[i-1] <= -100 and above_weekly_ema:
                    position = 1
                    signals[i] = 0.25
                # Short: CCI crosses below +100 in downtrend
                elif cci[i] < 100 and cci[i-1] >= 100 and below_weekly_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals