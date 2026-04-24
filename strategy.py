#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA trend filter and volume confirmation.
- Primary timeframe: 6h for signals.
- HTF: 12h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 on 6h.
- Entry: Long when Bull Power > 0 AND 12h trend bullish AND volume > 1.5 * 20-period 6h volume MA.
         Short when Bear Power < 0 AND 12h trend bearish AND volume > 1.5 * 20-period 6h volume MA.
- Exit: Opposite Elder Ray condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear strength relative to EMA, working in both trends and ranges when combined with volume and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray on 6h
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34
    df_12h_close = df_12h['close'].values
    ema34_12h = pd.Series(df_12h_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # 12h trend: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend_12h = np.where(df_12h_close > ema34_12h, 1, np.where(df_12h_close < ema34_12h, -1, 0))
    
    # Calculate 20-period volume MA on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 12h volume MA (aligned)
    volume_confirm = volume > (1.5 * vol_ma_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)  # Need enough bars for EMA13, EMA34, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        trend_val = trend_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if vol_conf:
                # Bullish: Bull Power > 0 AND 12h trend bullish
                if curr_bull > 0 and trend_val == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bear Power < 0 AND 12h trend bearish
                elif curr_bear < 0 and trend_val == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR loss of volume confirmation OR trend turns bearish
            if curr_bull <= 0 or not vol_conf or trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR loss of volume confirmation OR trend turns bullish
            if curr_bear >= 0 or not vol_conf or trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA13_12hEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0