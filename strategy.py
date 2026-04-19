#!/usr/bin/env python3
"""
1d_Consolidation_Breakout_With_Trend_Filter
Hypothesis: Daily consolidation breakouts with weekly trend filter work in both bull and bear markets.
Consolidation is identified by low ATR volatility, breakouts by price moving outside Bollinger Bands.
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
Works in bull/bear via weekly trend filter and volatility-based entry conditions.
"""

name = "1d_Consolidation_Breakout_With_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d indicators
    # ATR for volatility measurement and consolidation detection
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.zeros_like(high)
        atr[:period-1] = np.nan
        if len(high) >= period:
            atr[period-1] = np.nanmean(tr[:period])
            for i in range(period, len(high)):
                if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                    atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                else:
                    atr[i] = np.nan
        return atr
    
    # Bollinger Bands
    def calculate_bollinger_bands(close, period=20, std_dev=2.0):
        sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
        std = pd.Series(close).rolling(window=period, min_periods=period).std().values
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, lower, sma
    
    # Calculate 1d ATR and Bollinger Bands
    atr_1d = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, 20, 2.0)
    
    # Consolidation condition: low volatility (ATR below its 50-period average)
    atr_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    consolidation = atr_1d < (atr_ma * 0.7)  # ATR significantly below average
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d[i]) or np.isnan(atr_ma[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Consolidation breakout conditions
        is_consolidating = consolidation[i]
        price_above_upper = close[i] > bb_upper[i]
        price_below_lower = close[i] < bb_lower[i]
        
        # Weekly trend direction
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: consolidation breakout above upper BB in weekly uptrend
            if (is_consolidating and price_above_upper and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: consolidation breakout below lower BB in weekly downtrend
            elif (is_consolidating and price_below_lower and weekly_downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below middle BB or consolidation ends
            if (close[i] < bb_middle[i]) or (not consolidation[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above middle BB or consolidation ends
            if (close[i] > bb_middle[i]) or (not consolidation[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals