#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot R1/S1 Breakout with 4h Trend Filter and Session Filter.
Long when price breaks above R1 AND 4h close > 4h open (bullish 4h candle).
Short when price breaks below S1 AND 4h close < 4h open (bearish 4h candle).
Exit when price returns to Camarilla pivot point (PP) or opposite session.
Uses 4h for trend direction (bullish/bearish candle), 1h for Camarilla calculation and entry timing.
Session filter: 08-20 UTC to avoid low-volume periods.
Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

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
    
    # Get 4h data for trend filter (bullish/bearish candle)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Calculate 4h candle direction: 1 for bullish (close > open), -1 for bearish (close < open), 0 for doji
    bullish_4h = close_4h > open_4h
    bearish_4h = close_4h < open_4h
    candle_dir_4h = np.where(bullish_4h, 1, np.where(bearish_4h, -1, 0))
    
    # Align 4h candle direction to 1h timeframe
    candle_dir_4h_aligned = align_htf_to_ltf(prices, df_4h, candle_dir_4h)
    
    # Calculate Camarilla pivots on 1h timeframe using previous bar's OHLC
    # Camarilla equations:
    # PP = (High + Low + Close) / 3
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    # We need previous bar's data to avoid look-ahead
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3.0
    r1 = np.roll(close, 1) + (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 12.0
    s1 = np.roll(close, 1) - (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 12.0
    
    # Set first bar to NaN since we don't have previous bar
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 1  # warmup for pivot calculation
    
    # Pre-compute session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    for i in range(start_idx, n):
        # Skip if required data is not available
        if np.isnan(pp[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(candle_dir_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        pp_val = pp[i]
        r1_val = r1[i]
        s1_val = s1[i]
        trend = candle_dir_4h_aligned[i]  # 1=bullish, -1=bearish, 0=doji
        
        if position == 0:
            # Long: price breaks above R1 AND 4h candle is bullish
            if price > r1_val and trend == 1:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND 4h candle is bearish
            elif price < s1_val and trend == -1:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point (PP) or 4h candle turns bearish
            if price <= pp_val or trend == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to pivot point (PP) or 4h candle turns bullish
            if price >= pp_val or trend == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hCandleTrend_Session"
timeframe = "1h"
leverage = 1.0