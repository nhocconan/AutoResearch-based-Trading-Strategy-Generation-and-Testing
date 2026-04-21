# %%
#!/usr/bin/env python3
"""
1d_1w_CCI_MeanReversion_WithTrendFilter
Hypothesis: On 1d timeframe, CCI identifies overbought/oversold conditions. 
Only take mean-reversion trades when aligned with 1w trend (price above/below 200 EMA on weekly).
Works in bull/bear by fading extremes only in direction of higher timeframe trend.
Uses weekly timeframe for trend filter, daily for signal generation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 200 EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily price data for CCI calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate CCI(20) on daily data
    typical_price = (high + low + close) / 3.0
    
    # Moving average of typical price
    ma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    
    # Mean deviation
    md = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    
    # CCI calculation
    cci = (typical_price - ma_tp) / (0.015 * md)
    # Handle division by zero or near-zero mean deviation
    cci = np.where(md == 0, 0, cci)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(cci[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_200 = ema_200_1w_aligned[i]
        cci_value = cci[i]
        
        # Trend filter: price above/below weekly 200 EMA
        price_above_ema = price > ema_200
        price_below_ema = price < ema_200
        
        if position == 0:
            # Long: oversold (CCI < -100) + price above weekly 200 EMA (uptrend)
            if cci_value < -100 and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: overbought (CCI > 100) + price below weekly 200 EMA (downtrend)
            elif cci_value > 100 and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI returns to neutral zone or trend reversal
            if cci_value > -50 or price < ema_200:  # Exit on recovery or trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI returns to neutral zone or trend reversal
            if cci_value < 50 or price > ema_200:  # Exit on recovery or trend break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_CCI_MeanReversion_WithTrendFilter"
timeframe = "1d"
leverage = 1.0
# %%