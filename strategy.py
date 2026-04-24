#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend filter (bullish if close > EMA50, bearish if close < EMA50).
- Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13).
- Regime: Only trade Bull Power breakouts in bullish 12h regime, Bear Power breakdowns in bearish regime.
- Volume filter: Require volume > 1.5 * 20-period average volume for confirmation.
- Entry: Long when Bull Power crosses above zero AND bullish regime AND volume confirmation.
         Short when Bear Power crosses below zero AND bearish regime AND volume confirmation.
- Exit: Opposite signal (Long exits when Bear Power < 0, Short exits when Bull Power > 0).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via Bull Power longs and bear markets via Bear Power shorts,
  while avoiding whipsaws via regime and volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h trend: bullish if close > EMA50, bearish if close < EMA50
    # We need the 12h close aligned to 6h
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    bullish_regime = close_12h_aligned > ema50_12h_aligned
    bearish_regime = close_12h_aligned < ema50_12h_aligned
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Need 13 for EMA13, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Exit conditions: opposite Elder Ray signal
        if position != 0:
            # Exit long: Bear Power < 0 (momentum fading)
            if position == 1:
                if bear_power[i] < 0:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Bull Power > 0 (momentum fading)
            elif position == -1:
                if bull_power[i] > 0:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Elder Ray crossover with regime and volume filters
        if position == 0:
            # Long: Bull Power crosses above zero AND bullish regime AND volume confirmation
            long_condition = (bull_power[i] > 0 and bull_power[i-1] <= 0 and
                            bullish_regime[i] and
                            volume_confirm)
            
            # Short: Bear Power crosses below zero AND bearish regime AND volume confirmation
            short_condition = (bear_power[i] < 0 and bear_power[i-1] >= 0 and
                             bearish_regime[i] and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0