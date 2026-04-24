#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA(34) trend filter and 6h volume spike confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 6h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA on 6h).
- Entry: Long when Bull Power > 0 AND 1d EMA34 trend bullish (close > EMA34) AND volume spike.
         Short when Bear Power < 0 AND 1d EMA34 trend bearish (close < EMA34) AND volume spike.
- Exit: Opposite Elder Ray signal (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit) OR loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA(13) for Elder Ray on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Get 1d data for EMA(34) trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA(34) to 6h (with completed-bar delay)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA on 6h
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period volume MA
    volume_spike = volume > (2.0 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # Need enough bars for EMA34, volume MA, and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish entry: Bull Power > 0 AND 1d EMA34 trend bullish (close > EMA34)
                if bull_power[i] > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 AND 1d EMA34 trend bearish (close < EMA34)
                elif bear_power[i] < 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 OR loss of volume confirmation
            if bull_power[i] <= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 OR loss of volume confirmation
            if bear_power[i] >= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA34Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0