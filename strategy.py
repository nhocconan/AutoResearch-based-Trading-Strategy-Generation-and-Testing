#!/usr/bin/env python3
"""
4h_cci_volume_regime_v1
Hypothesis: On 4h timeframe, enter long when CCI(20) crosses above -100 with volume > 1.5x average, enter short when CCI(20) crosses below +100 with volume > 1.5x average. Uses 1d trend filter (price above/below 50-period EMA) to avoid counter-trend trades. Designed for 20-40 trades/year to minimize fee drag while capturing momentum reversals in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_cci_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate CCI(20)
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 0.001, mad)
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(cci[i]) or np.isnan(cci[i-1]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: CCI crosses below +100 (momentum fading)
            if cci[i] < 100 and cci[i-1] >= 100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above -100 (momentum fading)
            if cci[i] > -100 and cci[i-1] <= -100:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: CCI crosses above -100 (bullish momentum) + price > 1d EMA50
                if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: CCI crosses below +100 (bearish momentum) + price < 1d EMA50
                elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals