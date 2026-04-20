#!/usr/bin/env python3
# 6h_WeeklyPivot_Trend_Scalper
# Hypothesis: Weekly pivot levels act as strong support/resistance. Price tends to trend away from
# weekly pivot (PP) with momentum. In bull markets, price stays above PP; in bear markets, below PP.
# We go long when price crosses above weekly PP with bullish momentum (close > open) and short
# when price crosses below weekly PP with bearish momentum (close < open). Uses 6h timeframe to
# reduce noise and frequency. Volatility filter ensures trades only in sufficient volatility.

name = "6h_WeeklyPivot_Trend_Scalper"
timeframe = "6h"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    wh = df_1w['high'].values
    wl = df_1w['low'].values
    wc = df_1w['close'].values
    
    # Weekly pivot: PP = (H + L + C) / 3
    pp = (wh + wl + wc) / 3
    
    # Align weekly PP to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    
    # Momentum filter: bullish/bearish candle
    bullish = close > open_price
    bearish = close < open_price
    
    # Volatility filter: ATR > 0.5 * ATR(50)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma50 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Ensure ATR and momentum are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price crosses above weekly PP with bullish candle + volatility
            if close[i] > pp_aligned[i] and close[i-1] <= pp_aligned[i-1] and bullish[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly PP with bearish candle + volatility
            elif close[i] < pp_aligned[i] and close[i-1] >= pp_aligned[i-1] and bearish[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses back below weekly PP
            if close[i] < pp_aligned[i] and close[i-1] >= pp_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses back above weekly PP
            if close[i] > pp_aligned[i] and close[i-1] <= pp_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals