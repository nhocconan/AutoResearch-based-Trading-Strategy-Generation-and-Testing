#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 6h for Bull Power/Bear Power calculations.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 6h volume > 1.8 * 20-period 6h volume MA to confirm conviction.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low.
- Entry: Long when Bull Power > 0 AND 1d EMA34 trend bullish AND volume spike.
         Short when Bear Power > 0 AND 1d EMA34 trend bearish AND volume spike.
- Exit: Opposite power condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Elder Ray measures bull/bear power relative to EMA13, working in both bull and bear markets by
focusing on institutional buying/selling pressure with trend and volume filters.
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
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = EMA13 - Low
    bear_power = ema13 - low
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 6h volume MA
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_6h)
    
    # Get 1d data for EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    df_1d_ema34 = pd.Series(df_1d_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # 1d trend: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend_1d = np.where(df_1d_close > df_1d_ema34, 1, np.where(df_1d_close < df_1d_ema34, -1, 0))
    
    # Align HTF indicators to 6h
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(trend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Bull Power > 0 AND 1d EMA34 trend bullish
                if bull_power[i] > 0 and trend_1d_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Bear Power > 0 AND 1d EMA34 trend bearish
                elif bear_power[i] > 0 and trend_1d_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR loss of volume confirmation OR trend turns bearish
            if bull_power[i] <= 0 or not volume_spike[i] or trend_1d_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power <= 0 OR loss of volume confirmation OR trend turns bullish
            if bear_power[i] <= 0 or not volume_spike[i] or trend_1d_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0