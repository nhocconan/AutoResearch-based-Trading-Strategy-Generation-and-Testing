#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX(14) for regime (trending when ADX > 25, ranging when ADX < 20) with hysteresis.
- Entry: In trending regime (ADX > 25), go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
          In ranging regime (ADX < 20), fade extremes: long when Bear Power < -std and turning up, short when Bull Power > std and turning down.
- Exit: Opposite signal or ATR-based stop (1.5 * ATR(14)).
- Signal size: 0.25 discrete to balance capture and fee control.
- Volume confirmation: current volume > 1.5 * 20-period volume MA.
Designed to work in both bull and bear markets by adapting to regime: trend following in strong trends, mean reversion in ranges.
Elder Ray measures bull/bear power relative to EMA13, providing clear directional signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray calculations
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Calculate 12h ADX(14) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h ATR(14) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr2_6h[0] = 0
    tr3_6h[0] = 0
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate standard deviation of Bear/Bull Power for ranging signals
    bull_power_std = pd.Series(bull_power).rolling(window=50, min_periods=50).std().values
    bear_power_std = pd.Series(bear_power).rolling(window=50, min_periods=50).std().values
    bull_power_std_aligned = align_htf_to_ltf(prices, df_6h, bull_power_std)
    bear_power_std_aligned = align_htf_to_ltf(prices, df_6h, bear_power_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr_6h_aligned[i]) or 
            np.isnan(vol_ma_6h_aligned[i]) or np.isnan(bull_power_std_aligned[i]) or
            np.isnan(bear_power_std_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Regime determination with hysteresis
        adx_val = adx_aligned[i]
        if adx_val > 25:
            regime = 'trending'
        elif adx_val < 20:
            regime = 'ranging'
        else:
            regime = regime  # maintain previous regime
        
        # Volume confirmation
        vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        if position == 0:
            if regime == 'trending':
                # Trend following: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
                bull_power_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
                bear_power_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
                
                if bull_power_aligned[i] > 0 and bull_power_rising and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif bear_power_aligned[i] < 0 and bear_power_falling and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
            else:  # ranging regime
                # Mean reversion: fade extremes
                bear_power_extreme = bear_power_aligned[i] < -1.0 * bear_power_std_aligned[i]
                bull_power_extreme = bull_power_aligned[i] > 1.0 * bull_power_std_aligned[i]
                bear_power_turning_up = bear_power_aligned[i] > bear_power_aligned[i-1]
                bull_power_turning_down = bull_power_aligned[i] < bull_power_aligned[i-1]
                
                if bear_power_extreme and bear_power_turning_up and vol_confirmed:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                elif bull_power_extreme and bull_power_turning_down and vol_confirmed:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or Bear Power > 0 (trend exhaustion)
            stop_loss = entry_price - 1.5 * atr_6h_aligned[i]
            if curr_low < stop_loss or bear_power_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or Bull Power < 0 (trend exhaustion)
            stop_loss = entry_price + 1.5 * atr_6h_aligned[i]
            if curr_high > stop_loss or bull_power_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADXRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0