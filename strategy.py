#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d/1w regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA trend, 1w for higher-timeframe trend filter.
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) (using 13-period EMA on 6h).
- Regime: 1d price > 1d EMA50 = bullish regime (favor longs), 1d price < 1d EMA50 = bearish regime (favor shorts).
          Additionally, require 1w price > 1w EMA34 for long bias, 1w price < 1w EMA34 for short bias.
- Entry: Long when Bull Power > 0 AND 1d regime bullish AND 1w regime bullish AND volume > 1.5 * 20-period volume MA.
         Short when Bear Power < 0 AND 1d regime bearish AND 1w regime bearish AND volume > 1.5 * 20-period volume MA.
- Exit: Opposite Elder Ray signal (Bull Power < 0 for long exit, Bear Power > 0 for short exit) OR regime shift.
- Volume confirmation: avoids low-volume false signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
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
    
    # Get 1d data for EMA50 (trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA34 (higher-timeframe trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w EMA34
    close_1w = pd.Series(df_1w['close'])
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d and 1w EMAs to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Elder Ray: 13-period EMA on 6h close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 34, 13, 20)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema50_val = ema50_1d_aligned[i]
        ema34_val = ema34_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Long conditions: Bull Power > 0 AND 1d bullish regime AND 1w bullish regime
                if bull_val > 0 and curr_close > ema50_val and curr_close > ema34_val:
                    signals[i] = 0.25
                    position = 1
                # Short conditions: Bear Power < 0 AND 1d bearish regime AND 1w bearish regime
                elif bear_val < 0 and curr_close < ema50_val and curr_close < ema34_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 OR regime shifts to bearish (price < 1d EMA50 OR price < 1w EMA34)
            if bull_val < 0 or curr_close < ema50_val or curr_close < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power > 0 OR regime shifts to bullish (price > 1d EMA50 OR price > 1w EMA34)
            if bear_val > 0 or curr_close > ema50_val or curr_close > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1d1wEMA_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0