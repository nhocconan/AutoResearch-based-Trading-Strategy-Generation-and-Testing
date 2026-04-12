#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_cci_reversal_v1
# Commodity Channel Index (CCI) reversal strategy on 4h with 1d trend filter.
# In bull markets: buy when CCI crosses above -100 from oversold, in 1d uptrend.
# In bear markets: sell when CCI crosses below +100 from overbought, in 1d downtrend.
# Uses 1d EMA50 as trend filter to avoid counter-trend trades.
# Volume confirmation required to avoid false signals.
# Target: 20-40 trades/year per symbol for low friction.
name = "4h_1d_cci_reversal_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate CCI(20) on 4h
    typical_price = (high + low + close) / 3.0
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    mad = np.where(mad == 0, 1e-10, mad)
    cci = (typical_price - ma_tp) / (0.015 * mad)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after warmup
        # Skip if indicators not ready
        if np.isnan(cci[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_confirm[i]):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: CCI crosses above -100 from oversold AND 1d uptrend (price > EMA50)
        if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema50_1d_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: CCI crosses below +100 from overbought AND 1d downtrend (price < EMA50)
        elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema50_1d_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite CCI cross
        elif cci[i] < -100 and cci[i-1] >= -100 and position == 1:
            position = 0
            signals[i] = 0.0
        elif cci[i] > 100 and cci[i-1] <= 100 and position == -1:
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