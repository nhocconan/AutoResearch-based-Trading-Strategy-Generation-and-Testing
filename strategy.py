#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ADX(14) for regime (trending when ADX > 25, ranging when ADX < 20).
- Entry: Long when Bull Power > 0 AND EMA13(close) rising AND volume > 1.5 * 20-volume MA in trending regime.
         Short when Bear Power < 0 AND EMA13(close) falling AND volume > 1.5 * 20-volume MA in trending regime.
- Exit: Opposite Elder Ray signal or volume drops below average.
- Signal size: 0.25 discrete to balance capture and fee control.
Elder Ray measures bull/bear power relative to EMA13, working in both bull and bear markets by capturing strength/weakness.
ADX regime filter avoids false signals in ranging markets. Volume confirmation reduces false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for EMA13 and Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h EMA13 for Elder Ray
    close_6h = df_6h['close'].values
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    bull_power = high_6h - ema13
    bear_power = low_6h - ema13
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_6h, ema13)
    
    # Calculate 6h EMA13 slope for trend confirmation (rising/falling)
    ema13_slope = np.diff(ema13, prepend=ema13[0])
    ema13_slope_aligned = align_htf_to_ltf(prices, df_6h, ema13_slope)
    
    # Calculate 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # positive down move
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    atr_1d = wilders_smoothing(tr_1d, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / (atr_1d + 1e-10)
    minus_di_14 = 100 * minus_dm_14 / (atr_1d + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 30, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_slope_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime: trending when ADX > 25, ranging when ADX < 20 (with hysteresis)
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        if position == 0:
            # Volume confirmation: 1.5x average volume
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            
            # EMA13 slope: rising when positive, falling when negative
            ema13_rising = ema13_slope_aligned[i] > 0
            ema13_falling = ema13_slope_aligned[i] < 0
            
            # Long: Bull Power > 0 AND EMA13 rising AND volume confirmed AND trending regime
            if bull_power_aligned[i] > 0 and ema13_rising and vol_confirmed and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power < 0 AND EMA13 falling AND volume confirmed AND trending regime
            elif bear_power_aligned[i] < 0 and ema13_falling and vol_confirmed and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on Bear Power < 0 OR volume drops below average OR ADX falls to ranging
            vol_normal = curr_volume < vol_ma_6h_aligned[i]  # volume below average
            if bear_power_aligned[i] < 0 or vol_normal or (adx_aligned[i] < 20 and is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Bull Power > 0 OR volume drops below average OR ADX falls to ranging
            vol_normal = curr_volume < vol_ma_6h_aligned[i]  # volume below average
            if bull_power_aligned[i] > 0 or vol_normal or (adx_aligned[i] < 20 and is_trending):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0