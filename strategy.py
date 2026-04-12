#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_12h_1d_cci_reversal_v1
# Uses CCI(20) on 12h timeframe to detect overbought/oversold conditions.
# Combines with 1d trend filter (price > EMA50 for long, price < EMA50 for short) to align with higher timeframe trend.
# Entry: CCI crosses below -100 (oversold) in uptrend (price > EMA50) for long.
# Entry: CCI crosses above +100 (overbought) in downtrend (price < EMA50) for short.
# Exit: CCI returns to neutral zone (-100 to +100) or trend reversal.
# Designed for low trade frequency (12-30/year) with high edge in ranging markets with trend bias.

name = "6h_12h_1d_cci_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for CCI calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate CCI(20) on 12h
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    tp_ma = typical_price.rolling(window=20, min_periods=20).mean()
    tp_mad = typical_price.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - tp_ma) / (0.015 * tp_mad)
    cci_values = cci.values
    cci_12h_aligned = align_htf_to_ltf(prices, df_12h, cci_values)
    
    # Calculate EMA50 on 1d for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if values not ready
        if np.isnan(cci_12h_aligned[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        cci_val = cci_12h_aligned[i]
        price = close[i]
        ema50 = ema_50_aligned[i]
        
        # Determine trend: above EMA50 = uptrend, below = downtrend
        uptrend = price > ema50
        downtrend = price < ema50
        
        # Long entry: CCI crosses below -100 (oversold) in uptrend
        long_entry = (cci_val < -100) and uptrend and (position != 1)
        # Short entry: CCI crosses above +100 (overbought) in downtrend
        short_entry = (cci_val > 100) and downtrend and (position != -1)
        
        # Exit: CCI returns to neutral zone (-100 to +100)
        exit_long = position == 1 and (-100 <= cci_val <= 100)
        exit_short = position == -1 and (-100 <= cci_val <= 100)
        
        if long_entry:
            position = 1
            signals[i] = 0.25
        elif short_entry:
            position = -1
            signals[i] = -0.25
        elif exit_long or exit_short:
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