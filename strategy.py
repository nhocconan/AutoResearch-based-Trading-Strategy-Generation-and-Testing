#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA(34) trend filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1d EMA(34) for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h).
- Volume: Current 6h volume > 1.8 * 20-period volume MA to avoid low-momentum breakouts.
- Entry: Long when Bull Power > 0 AND 1d EMA34 trend bullish (close > EMA34) AND volume spike.
         Short when Bear Power < 0 AND 1d EMA34 trend bearish (close < EMA34) AND volume spike.
- Exit: Opposite Elder Ray signal (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Why should work in both bull and bear: Elder Ray measures buying/selling pressure relative to trend,
  volume confirms conviction, and 1d EMA34 filters counter-trend noise. Works in trends (strong Elder Ray)
  and ranges (reversions at extremes).
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
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying power
    bear_power = low - ema13   # Selling power (negative values indicate selling pressure)
    
    # Get 1d data for EMA(34) trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d close
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 13, 20)  # Need enough 1d bars for EMA34 and volume MA, plus EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_34_val = ema_34_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish entry: Bull Power > 0 (buying pressure) AND 1d EMA34 bullish (close > EMA34)
                if bull_power[i] > 0 and curr_close > ema_34_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bear Power < 0 (selling pressure) AND 1d EMA34 bearish (close < EMA34)
                elif bear_power[i] < 0 and curr_close < ema_34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 (buying pressure faded) OR Bear Power >= 0 (selling pressure emerged)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 (selling pressure faded) OR Bull Power > 0 (buying pressure emerged)
            if bear_power[i] >= 0 or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMA13_1dEMA34Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0