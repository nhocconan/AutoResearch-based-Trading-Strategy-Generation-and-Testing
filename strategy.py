#!/usr/bin/env python3
"""
1h_VWAP_Bounce_4hTrend_Filter
Hypothesis: Price bouncing off VWAP on 1h with confluence from 4h trend (EMA50) and volume spike (1.5x median) captures mean-reversion in range-bound markets while avoiding false signals. Works in both bull and bear markets by only taking long positions in 4h uptrends and short positions in 4h downtrends. Target: 20-50 trades/year to avoid fee drag.
"""

name = "1h_VWAP_Bounce_4hTrend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h VWAP calculation
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    vwap_numerator = (typical_price * prices['volume']).cumsum()
    vwap_denominator = prices['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: spike above 1.5x median of last 20 periods
    vol_median = prices['volume'].rolling(window=20, min_periods=10).median().values
    vol_threshold = vol_median * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap.iloc[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            if position != 0:
                # Simple exit: price crosses VWAP in opposite direction
                if position == 1 and prices['close'].iloc[i] < vwap.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and prices['close'].iloc[i] > vwap.iloc[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20 if position == 1 else -0.20
            continue
        
        # Determine 4h trend
        trend_up = prices['close'].iloc[i] > ema50_4h_aligned[i]
        trend_down = prices['close'].iloc[i] < ema50_4h_aligned[i]
        
        # Volume filter: spike above 1.5x median
        vol_ok = prices['volume'].iloc[i] > vol_threshold[i]
        
        # Price relative to VWAP
        price_above_vwap = prices['close'].iloc[i] > vwap.iloc[i]
        price_below_vwap = prices['close'].iloc[i] < vwap.iloc[i]
        
        if position == 0:
            # Look for entries: price at VWAP with 4h trend and volume spike
            if price_below_vwap and trend_up and vol_ok:
                # Long: price at/below VWAP + 4h uptrend + volume spike
                signals[i] = 0.20
                position = 1
                entry_price = prices['close'].iloc[i]
            elif price_above_vwap and trend_down and vol_ok:
                # Short: price at/above VWAP + 4h downtrend + volume spike
                signals[i] = -0.20
                position = -1
                entry_price = prices['close'].iloc[i]
        else:
            # Exit: price crosses VWAP in opposite direction
            if position == 1 and prices['close'].iloc[i] < vwap.iloc[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and prices['close'].iloc[i] > vwap.iloc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals