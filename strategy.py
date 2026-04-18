#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_Volume_Trend_Filter
Hypothesis: Price breaks above/below the Camarilla R1/S1 pivot levels on 1h with volume confirmation and 4h EMA trend filter.
Uses 4h EMA34 for trend direction and 1h volume > 1.5x average for confirmation.
Designed to capture volatility expansion moves in both bull and bear markets with tight entry conditions.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h calculations
    # Camarilla pivot levels (based on previous day's OHLC)
    # Since we don't have daily data in 1h timeframe, we'll use 4h data for pivots
    # But for 1h Camarilla, we need 1h OHLC of previous day - approximated via rolling
    # Use 24-period (1 day of 1h bars) for high/low/close
    if len(high) >= 24:
        # Previous day's OHLC (using 24-period rolling, shifted by 1 to avoid lookahead)
        prev_day_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
        prev_day_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
        prev_day_close = pd.Series(close).rolling(window=24, min_periods=24).mean().shift(1).values
        
        # Camarilla calculations
        rang = prev_day_high - prev_day_low
        r1 = prev_day_close + rang * 1.1 / 12
        s1 = prev_day_close - rang * 1.1 / 12
    else:
        # Not enough data
        r1 = np.full(n, np.nan)
        s1 = np.full(n, np.nan)
    
    # Volume filter: >1.5x 24-period average (1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # 4h EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 34:
        ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    else:
        ema_34_4h_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 25  # Need 24 for lookback + 1
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_34_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1[i]
        s1_level = s1[i]
        vol_ok = volume_filter[i]
        ema34 = ema_34_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend (price > 4h EMA34)
            if price > r1_level and vol_ok and price > ema34:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume in downtrend (price < 4h EMA34)
            elif price < s1_level and vol_ok and price < ema34:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: price returns to S1 or trend reverses
            if price < s1_level or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: price returns to R1 or trend reverses
            if price > r1_level or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_Volume_Trend_Filter"
timeframe = "1h"
leverage = 1.0