#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX(14) for regime (trending when ADX > 25, ranging when ADX < 20).
- Entry: Long when Bull Power > 0 AND ADX > 25 AND volume > 1.5 * 6h volume MA(20); Short when Bear Power < 0 AND ADX > 25 AND volume confirmed.
- Exit: Position closes when Elder Ray power reverses sign or ADX drops below 20 (regime change).
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull markets by capturing strong uptrends with Bull Power, works in bear markets by capturing strong downtrends with Bear Power.
- Avoids whipsaws in ranging markets via ADX < 20 filter.
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
    
    # Get 6h data for Elder Ray and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 6h EMA(13) for Elder Ray (standard setting)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA(13)
    bear_power = low - ema_13   # Bear Power: Low - EMA(13)
    
    # Calculate 12h ADX(14) for regime
    # ADX requires +DI, -DI, and DX calculations
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range (TR)
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = (prev_smoothed * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    plus_dm_14 = wilders_smoothing(plus_dm, 14)
    minus_dm_14 = wilders_smoothing(minus_dm, 14)
    
    # Calculate +DI and -DI
    plus_di = np.where(tr_14 != 0, (plus_dm_14 / tr_14) * 100, 0)
    minus_di = np.where(tr_14 != 0, (minus_dm_14 / tr_14) * 100, 0)
    
    # Calculate DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilders_smoothing(dx, 14)  # ADX is smoothed DX
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 30)  # Enough for EMA13, ADX, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and ADX > 25 (trending regime)
            vol_confirmed = curr_volume > 1.5 * vol_ma_6h_aligned[i]
            strong_trend = adx_aligned[i] > 25
            
            # Long: Bull Power > 0 AND strong trend AND volume confirmed
            if bull_power_aligned[i] > 0 and strong_trend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND strong trend AND volume confirmed
            elif bear_power_aligned[i] < 0 and strong_trend and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when Bull Power <= 0 (momentum fading) OR ADX < 20 (regime change to ranging)
            if bull_power_aligned[i] <= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Bear Power >= 0 (momentum fading) OR ADX < 20 (regime change to ranging)
            if bear_power_aligned[i] >= 0 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_12hADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0