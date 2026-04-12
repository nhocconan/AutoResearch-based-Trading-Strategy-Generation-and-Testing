#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_cci_trend_v1
# Uses daily CCI(20) as trend filter and 4h CCI(14) for entry timing.
# Long when daily CCI > 0 and 4h CCI crosses above -100 (mean reversion in uptrend).
# Short when daily CCI < 0 and 4h CCI crosses below 100 (mean reversion in downtrend).
# Volume confirmation (volume > 1.3x 20-period average) to filter false signals.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).

name = "4h_1d_cci_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI(20)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    sma_tp_1d = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).mean().values
    mad_1d = pd.Series(typical_price_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (typical_price_1d - sma_tp_1d) / (0.015 * mad_1d)
    cci_1d = np.where(mad_1d == 0, 0, cci_1d)  # avoid division by zero
    
    # Align daily CCI to 4h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Calculate 4h CCI(14) for entry timing
    typical_price = (high + low + close) / 3
    sma_tp = pd.Series(typical_price).rolling(window=14, min_periods=14).mean().values
    mad = pd.Series(typical_price).rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    cci = np.where(mad == 0, 0, cci)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(cci_1d_aligned[i]) or np.isnan(cci[i]) or np.isnan(sma_tp[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: daily CCI positive AND 4h CCI crosses above -100
        if cci_1d_aligned[i] > 0 and cci[i] > -100 and (i == 20 or cci[i-1] <= -100) and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: daily CCI negative AND 4h CCI crosses below 100
        elif cci_1d_aligned[i] < 0 and cci[i] < 100 and (i == 20 or cci[i-1] >= 100) and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite daily CCI signal
        elif cci_1d_aligned[i] <= 0 and position == 1:
            position = 0
            signals[i] = 0.0
        elif cci_1d_aligned[i] >= 0 and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals