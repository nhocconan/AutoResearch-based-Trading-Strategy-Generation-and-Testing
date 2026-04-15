#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend)
# Short when Bear Power > 0 AND Bull Power < 0 AND 12h ADX > 25 (strong trend)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Elder Ray measures bull/bear power relative to EMA13, effective in both bull and bear markets.
# 12h ADX > 25 ensures we only trade during strong trends, avoiding whipsaws in ranging markets.
# Volume confirmation (1.5x 20-period average) ensures participation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # === 12h Indicator: ADX(14) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, 100 * dm_plus_smooth / atr_12h, 0)
    di_minus = np.where(atr_12h != 0, 100 * dm_minus_smooth / atr_12h, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_12h = wilders_smoothing(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === LTF Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 20, 14*3) + 5  # EMA13 + volume(20) + ADX smoothing + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (strong bulls)
        # 2. Bear Power < 0 (weak bears)
        # 3. 12h ADX > 25 (strong trend)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and \
           (bear_power[i] < 0) and \
           (adx_12h_aligned[i] > 25) and \
           vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power > 0 (strong bears)
        # 2. Bull Power < 0 (weak bulls)
        # 3. 12h ADX > 25 (strong trend)
        # 4. Volume confirmation
        elif (bear_power[i] > 0) and \
             (bull_power[i] < 0) and \
             (adx_12h_aligned[i] > 25) and \
             vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_12hADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0