#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Camarilla R4 level + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S4 level + 1d ADX > 25 (trending) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1d ADX provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-30 trades/year on 6h timeframe to avoid overtrading.
# Camarilla pivots provide mathematically derived support/resistance levels that work in ranging and trending markets.

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
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Camarilla Pivots (based on previous day) ===
    # Pivot = (High + Low + Close) / 3
    # R4 = Close + ((High - Low) * 1.1 / 2)
    # S4 = Close - ((High - Low) * 1.1 / 2)
    # Using previous day's data (shifted by 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First bar: use current values (no look-ahead)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    camarilla_r4 = camarilla_pivot + ((prev_high - prev_low) * 1.1 / 2.0)
    camarilla_s4 = camarilla_pivot - ((prev_high - prev_low) * 1.1 / 2.0)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # ADX(14) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Trend filter: ADX > 25 (trending market)
        trend_filter = adx_14_1d_aligned[i] > 25.0
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R4 level
        # 2. 1d ADX > 25 (trending)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           trend_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S4 level
        # 2. 1d ADX > 25 (trending)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             trend_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R4S4_1dADX_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0