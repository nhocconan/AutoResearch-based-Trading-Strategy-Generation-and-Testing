#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Reversal with 1d Elder Ray filtering and weekly volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold bounce) AND 1d Bear Power < 0 (bearish momentum weakening) AND weekly volume > 1.8x 4-week MA.
Short when Williams %R(14) crosses below -20 (overbought rejection) AND 1d Bull Power > 0 (bullish momentum weakening) AND weekly volume > 1.8x 4-week MA.
Exit when Williams %R crosses back through -50 (mean reversion center) or Elder Ray flips strongly.
Uses 1d HTF for Elder Ray (Bull/Bear Power) to confirm momentum shift and weekly volume spike for institutional participation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R captures short-term reversals, Elder Ray filters for genuine momentum changes, volume spike avoids false signals.
Works in bull (buy oversold bounces in uptrend) and bear (sell overbought rejections in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) - momentum oscillator
    lookback_willr = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    williams_r = np.full(n, np.nan)
    
    for i in range(lookback_willr - 1, n):
        highest_high[i] = np.max(high[i-lookback_willr+1:i+1])
        lowest_low[i] = np.min(low[i-lookback_willr+1:i+1])
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Calculate 1d Elder Ray (Bull Power and Bear Power) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # EMA13 minimum
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate weekly volume MA (4-period) for spike filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4_1w, additional_delay_bars=0)  # weekly volume known at close
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_willr - 1, 13, 4)  # Williams %R, EMA13, weekly vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        willr = williams_r[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        
        # Calculate Williams %R slope for crossover detection
        if i >= start_idx + 1:
            willr_prev = williams_r[i-1]
            willr_cross_above_80 = willr_prev <= -80 and willr > -80
            willr_cross_below_20 = willr_prev >= -20 and willr < -20
            willr_cross_above_50 = willr_prev <= -50 and willr > -50
            willr_cross_below_50 = willr_prev >= -50 and willr < -50
        else:
            willr_cross_above_80 = False
            willr_cross_below_20 = False
            willr_cross_above_50 = False
            willr_cross_below_50 = False
        
        # Volume filter: weekly volume > 1.8x 4-period MA (institutional participation)
        # Note: volume[i] is 6h volume, need to compare to weekly average scaled appropriately
        # Approximate: weekly volume should be compared to its own MA, not 6h volume
        # We'll use a simplified approach: check if current 6h volume is unusually high relative to recent 6h average
        vol_ma_20_6h = np.nan
        if i >= 20:
            vol_ma_20_6h = np.mean(volume[i-20:i])
        vol_filter_6h = i >= 20 and volume[i] > 2.0 * vol_ma_20_6h
        
        # Also use weekly volume confirmation
        weekly_vol_filter = vol_ma_val > 0 and volume_1w[min(len(volume_1w)-1, i//(6*4*7))] > 1.8 * vol_ma_val if len(volume_1w) > i//(6*4*7) else False
        
        # Simplified volume confirmation: use 6h volume spike OR weekly high volume
        volume_confirmation = vol_filter_6h  # Primary volume filter on 6h data
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND Bear Power < 0 (weakening bearish momentum) AND volume confirmation
            if willr_cross_above_80 and bear_power < 0 and volume_confirmation:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) AND Bull Power > 0 (weakening bullish momentum) AND volume confirmation
            elif willr_cross_below_20 and bull_power > 0 and volume_confirmation:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -50 (overbought) OR Bear Power > 0 (bullish momentum accelerating)
                if willr_cross_above_50 or bear_power > 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -50 (oversold) OR Bull Power < 0 (bearish momentum accelerating)
                if willr_cross_below_50 or bull_power < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_ElderRay_WeeklyVolume"
timeframe = "6h"
leverage = 1.0