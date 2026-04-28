#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h ADX trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND 12h ADX > 25 (strong trend) AND volume spike (>2.0x average)
# Short when Bear Power < 0 AND 12h ADX > 25 AND volume spike
# Works in both bull and bear markets by combining momentum (Elder Ray) with trend strength (ADX)
# Target: 12-37 trades/year via tight Elder Ray + ADX + volume conditions

name = "6h_ElderRay_BullBearPower_12hADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need sufficient data for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 12h timeframe
    high_12h = pd.Series(df_12h['high'])
    low_12h = pd.Series(df_12h['low'])
    close_12h = pd.Series(df_12h['close'])
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h.shift(1))
    tr3 = np.abs(low_12h - close_12h.shift(1))
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = np.where((high_12h - high_12h.shift(1)) > (low_12h.shift(1) - low_12h),
                       np.maximum(high_12h - high_12h.shift(1), 0), 0)
    dm_minus = np.where((low_12h.shift(1) - low_12h) > (high_12h - high_12h.shift(1)),
                        np.maximum(low_12h.shift(1) - low_12h, 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initial values (simple average)
    tr_14 = tr_12h.rolling(window=period, min_periods=period).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=period, min_periods=period).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=period, min_periods=period).sum().values
    
    # Wilder's smoothing: subsequent values
    tr_12h_smoothed = np.full_like(tr_12h, np.nan, dtype=float)
    dm_plus_12h_smoothed = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_12h_smoothed = np.full_like(dm_minus, np.nan, dtype=float)
    
    # Set first valid smoothed values
    tr_12h_smoothed[period-1] = tr_14[period-1]
    dm_plus_12h_smoothed[period-1] = dm_plus_14[period-1]
    dm_minus_12h_smoothed[period-1] = dm_minus_14[period-1]
    
    # Apply Wilder's smoothing
    for i in range(period, len(tr_12h)):
        tr_12h_smoothed[i] = tr_12h_smoothed[i-1] - (tr_12h_smoothed[i-1] / period) + tr_12h.iloc[i]
        dm_plus_12h_smoothed[i] = dm_plus_12h_smoothed[i-1] - (dm_plus_12h_smoothed[i-1] / period) + dm_plus.iloc[i]
        dm_minus_12h_smoothed[i] = dm_minus_12h_smoothed[i-1] - (dm_minus_12h_smoothed[i-1] / period) + dm_minus.iloc[i]
    
    # Calculate DI+ and DI-
    di_plus_12h = 100 * dm_plus_12h_smoothed / tr_12h_smoothed
    di_minus_12h = 100 * dm_minus_12h_smoothed / tr_12h_smoothed
    
    # Calculate DX and ADX
    dx_12h = 100 * np.abs(di_plus_12h - di_minus_12h) / (di_plus_12h + di_minus_12h)
    
    # Smooth DX to get ADX (Wilder's smoothing again)
    adx_12h = np.full_like(dx_12h, np.nan, dtype=float)
    # Initial ADX value (simple average of first 'period' DX values)
    dx_valid = ~np.isnan(dx_12h)
    if np.sum(dx_valid) >= period:
        first_adx_idx = np.where(dx_valid)[0][period-1] if np.sum(dx_valid) >= period else len(dx_12h)-1
        adx_12h[first_adx_idx] = np.nanmean(dx_12h[first_adx_idx-period+1:first_adx_idx+1])
        # Subsequent ADX values
        for i in range(first_adx_idx+1, len(dx_12h)):
            if not np.isnan(dx_12h[i]):
                adx_12h[i] = (adx_12h[i-1] * (period-1) + dx_12h[i]) / period
    
    # Align 12h ADX to 6h timeframe (completed 12h candles only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 30)  # Need sufficient history for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        adx_val = adx_12h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 AND ADX > 25 (strong uptrend) AND volume spike
            if bull_power[i] > 0 and adx_val > 25 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Bear Power < 0 AND ADX > 25 (strong downtrend) AND volume spike
            elif bear_power[i] < 0 and adx_val > 25 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or Bull Power turns negative
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            # Exit on stoploss or Bull Power <= 0 (momentum fading)
            if price < stop_loss or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or Bear Power turns positive
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            # Exit on stoploss or Bear Power >= 0 (momentum fading)
            if price > stop_loss or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals