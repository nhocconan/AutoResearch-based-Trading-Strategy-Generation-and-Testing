#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d ADX trend filter and volume confirmation
# Long when Lips > Teeth > Jaw (bullish alignment) + 1d ADX > 25 (strong trend) + volume > 1.5x 20-period avg
# Short when Lips < Teeth < Jaw (bearish alignment) + 1d ADX > 25 + volume confirmation
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Williams Alligator identifies trend initiation and continuation, effective in both bull and bear markets.
# 1d ADX > 25 ensures we only trade strong trends, reducing whipsaws in ranging markets.
# Volume confirmation adds conviction to breakouts. Target: ~20-40 trades/year on 4h timeframe.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: ADX(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First period has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing, alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smoothing(tr, period)
    dm_plus_smooth = wilder_smoothing(dm_plus, period)
    dm_minus_smooth = wilder_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    di_minus = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilder_smoothing(dx, period)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Williams Alligator ===
    # Jaw: Blue line, 13-period SMMA smoothed by 8 periods
    # Teeth: Red line, 8-period SMMA smoothed by 5 periods
    # Lips: Green line, 5-period SMMA smoothed by 3 periods
    # SMMA (Smoothed Moving Average) = Wilder's smoothing
    
    def smma(values, period):
        return wilder_smoothing(values, period)
    
    # Jaw: 13-period SMMA of median price, smoothed by 8
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = smma(jaw_raw, 8)
    
    # Teeth: 8-period SMMA of median price, smoothed by 5
    teeth_raw = smma(median_price, 8)
    teeth = smma(teeth_raw, 5)
    
    # Lips: 5-period SMMA of median price, smoothed by 3
    lips_raw = smma(median_price, 5)
    lips = smma(lips_raw, 3)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # Alligator components + ADX + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: Lips > Teeth > Jaw
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation
        if (lips[i] > teeth[i] > jaw[i]) and \
           (adx_1d_aligned[i] > 25) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: Lips < Teeth < Jaw
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation
        elif (lips[i] < teeth[i] < jaw[i]) and \
             (adx_1d_aligned[i] > 25) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsAlligator_1dADX_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0