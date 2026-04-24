#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 12h EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Volume: Current 6h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when Bull Power > 0 AND 12h close > EMA34 AND volume spike.
         Short when Bear Power < 0 AND 12h close < EMA34 AND volume spike.
- Exit: Opposite Elder Ray condition or loss of volume confirmation or trend reversal.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear strength relative to recent trend (EMA13), working in both bull and bear markets by filtering with 12h trend.
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
    
    # Calculate 13-period EMA for Elder Ray on 6h
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend direction
    df_12h_close = df_12h['close'].values
    ema34_12h = pd.Series(df_12h_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 6h for volume confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_ma_6h_aligned = align_htf_to_ltf(prices, prices, vol_ma_6h)  # self-align for same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema34_val = ema34_12h_aligned[i]
        vol_ma_val = vol_ma_6h_aligned[i]
        
        # Volume confirmation: current 6h volume > 2.0 * 20-period volume MA
        volume_spike = volume[i] > (2.0 * vol_ma_val)
        
        if position == 0:
            # Check for entry signals with volume spike and trend alignment
            if volume_spike:
                # Bullish: Bull Power > 0 AND 12h close > EMA34 (uptrend)
                if bull_power[i] > 0 and curr_close > ema34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bear Power < 0 AND 12h close < EMA34 (downtrend)
                elif bear_power[i] < 0 and curr_close < ema34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR trend reversal OR loss of volume confirmation
            if (bull_power[i] <= 0 or curr_close < ema34_val or not volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR trend reversal OR loss of volume confirmation
            if (bear_power[i] >= 0 or curr_close > ema34_val or not volume_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0