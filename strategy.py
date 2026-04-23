#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray (Bull/Bear Power) regime filter and 1w ATR-based volume confirmation.
- Williams %R(14): Long when crosses above -80 from below (oversold bounce), Short when crosses below -20 from above (overbought rejection)
- 1d Elder Ray: Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
  Regime: Bullish when Bull Power > 0 AND Bear Power < 0; Bearish when Bull Power < 0 AND Bear Power > 0
- 1w ATR(14) volume filter: Current 6h volume > 1.5x 1w average volume when 1w ATR(14) is expanding (ATR > prior ATR)
  This confirms momentum in direction of the weekly trend.
- Works in bull markets (catches oversold bounces in uptrend) and bear markets (catches overbought rejections in downtrend).
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    lookback_willr = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback_willr - 1, n):
        highest_high[i] = np.max(high[i-lookback_willr+1:i+1])
        lowest_low[i] = np.min(low[i-lookback_willr+1:i+1])
    
    willr = np.full(n, np.nan)
    for i in range(lookback_willr - 1, n):
        if highest_high[i] != lowest_low[i]:
            willr[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            willr[i] = -50  # neutral when range is zero
    
    # Calculate 1d Elder Ray (Bull/Bear Power) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema_13_1d  # High - EMA13
    bear_power_1d = ema_13_1d - low_1d   # EMA13 - Low
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1w ATR(14) and average volume for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # True Range for 1w ATR
    tr_1w = np.zeros(len(df_1w))
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr_1w[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_ma_1w = pd.Series(volume_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # ATR expansion: current ATR > prior ATR (momentum increasing)
    atr_expanding = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(atr_14_aligned[i]) and not np.isnan(atr_14_aligned[i-1]):
            atr_expanding[i] = atr_14_aligned[i] > atr_14_aligned[i-1]
    
    # Volume confirmation: current 6h volume > 1.5x 1w average volume when ATR expanding
    vol_filter = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(volume[i]) and not np.isnan(vol_ma_1w_aligned[i]) and atr_expanding[i]:
            vol_filter[i] = volume[i] > 1.5 * vol_ma_1w_aligned[i]
        elif not np.isnan(volume[i]) and not np.isnan(vol_ma_1w_aligned[i]):
            vol_filter[i] = volume[i] > vol_ma_1w_aligned[i]  # fallback: above average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_willr - 1, 13, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R signals with crossover detection
        willr_long_signal = False
        willr_short_signal = False
        
        if i >= start_idx + 1:
            willr_prev = willr[i-1]
            willr_curr = willr[i]
            # Long: crosses above -80 from below (oversold bounce)
            if willr_prev <= -80 and willr_curr > -80:
                willr_long_signal = True
            # Short: crosses below -20 from above (overbought rejection)
            if willr_prev >= -20 and willr_curr < -20:
                willr_short_signal = True
        
        # 1d Elder Ray regime filter
        bull_regime = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        bear_regime = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        if position == 0:
            # Long: Williams %R oversold bounce AND bullish regime AND volume confirmation
            if willr_long_signal and bull_regime and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought rejection AND bearish regime AND volume confirmation
            elif willr_short_signal and bear_regime and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses below -50 (momentum loss) OR regime turns bearish
                if i >= start_idx + 1:
                    willr_prev = willr[i-1]
                    willr_curr = willr[i]
                    if willr_prev >= -50 and willr_curr < -50:  # crosses below -50
                        exit_signal = True
                    elif not bull_regime:  # regime no longer bullish
                        exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses above -50 (momentum loss) OR regime turns bullish
                if i >= start_idx + 1:
                    willr_prev = willr[i-1]
                    willr_curr = willr[i]
                    if willr_prev <= -50 and willr_curr > -50:  # crosses above -50
                        exit_signal = True
                    elif not bear_regime:  # regime no longer bearish
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Regime_1wATRVolFilter"
timeframe = "6h"
leverage = 1.0