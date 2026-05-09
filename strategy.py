#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_ForceIndex_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA13 for trend
    ema_13_1w = pd.Series(df_1w['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    ema_13_6h_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13_6h_raw
    bear_power = low - ema_13_6h_raw
    
    # Force Index (13-period EMA of price change * volume)
    price_change = np.diff(close, prepend=close[0])
    fi_raw = price_change * volume
    fi_13 = pd.Series(fi_raw).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Wait for EMA13 and FI
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13_6h[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(fi_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly EMA13 slope (up if current > previous)
        weekly_up = ema_13_6h[i] > ema_13_6h[i-1]
        weekly_down = ema_13_6h[i] < ema_13_6h[i-1]
        
        if position == 0:
            # Long: Bull Power > 0 AND Force Index > 0 AND weekly up
            if (bull_power[i] > 0 and fi_13[i] > 0 and weekly_up):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Force Index < 0 AND weekly down
            elif (bear_power[i] < 0 and fi_13[i] < 0 and weekly_down):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Force Index <= 0
            if bull_power[i] <= 0 or fi_13[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Force Index >= 0
            if bear_power[i] >= 0 or fi_13[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals