#!/usr/bin/env python3
"""
6h_1d_cci_reversal_v1
Hypothesis: Use daily CCI overbought/oversold levels to identify mean reversion opportunities,
filtered by 12h trend direction (using EMA cross) to avoid counter-trend trades.
- Only trade when 12h EMA(21) > EMA(50) for longs, or < for shorts (trend filter)
- Enter long when daily CCI(20) crosses below -100 (oversold) and price > 12h EMA(20)
- Enter short when daily CCI(20) crosses above +100 (overbought) and price < 12h EMA(20)
- Exit when CCI returns to neutral zone (-100 to 100) or trend filter fails
- Target: 20-40 trades/year to avoid overtrading on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate daily CCI(20) - using daily data from 6h prices
    # We need to resample conceptually but will use price data directly with period adjustment
    # For 6h data, 4 bars = 1 day, so we use 80 period for 20-day equivalent
    period = 20
    typical_price = (high + low + close) / 3
    tp_ma = pd.Series(typical_price).rolling(window=period*4, min_periods=period*4).mean().values
    tp_dev = pd.Series(abs(typical_price - tp_ma)).rolling(window=period*4, min_periods=period*4).mean().values
    # Avoid division by zero
    tp_dev = np.where(tp_dev == 0, 0.001, tp_dev)
    cci = (typical_price - tp_ma) / (0.015 * tp_dev)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(21) and EMA(50) on 12h data
    ema_21 = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 6h timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Trend bullish when EMA21 > EMA50
    bullish_trend = ema_21_aligned > ema_50_aligned
    bearish_trend = ema_21_aligned < ema_50_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(cci[i]) or np.isnan(ema_21_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: CCI returns above -100 or trend turns bearish
            if cci[i] > -100 or bearish_trend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: CCI returns below +100 or trend turns bullish
            if cci[i] < 100 or bullish_trend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: bullish trend + CCI crosses below -100 (oversold)
            if (bullish_trend[i] and 
                cci[i-1] >= -100 and cci[i] < -100 and
                close[i] > ema_21_aligned[i]):  # Additional filter: price above short-term EMA
                position = 1
                signals[i] = 0.25
            # Short: bearish trend + CCI crosses above +100 (overbought)
            elif (bearish_trend[i] and 
                  cci[i-1] <= 100 and cci[i] > 100 and
                  close[i] < ema_21_aligned[i]):  # Additional filter: price below short-term EMA
                position = -1
                signals[i] = -0.25
    
    return signals