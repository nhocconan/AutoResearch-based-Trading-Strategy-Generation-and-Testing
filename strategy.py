#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level AND 1d ADX > 25 (strong trend) AND volume > 2.0x 20-period average.
# Short when price breaks below Camarilla S3 level AND 1d ADX > 25 (strong trend) AND volume > 2.0x 20-period average.
# Uses ATR(14) trailing stop (2.5x) for risk control.
# Camarilla levels provide precise support/resistance that work in ranging and trending markets.
# 1d ADX > 25 ensures we only trade during strong trending regimes, reducing whipsaws in ranging markets.
# Volume spike (>2.0x average) confirms institutional participation in breakouts.
# Target: 60-120 total trades over 4 years (15-30/year) on 4h.

name = "4h_Camarilla_R3S3_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1*(high-low)/2
    # S3 = close - 1.1*(high-low)/2
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate ADX(14) on 1d data for trend strength filter
    # ADX calculation requires +DI, -DI, and DX
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * EWMA(+DM) / EWMA(TR)
    # -DI = 100 * EWMA(-DM) / EWMA(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # Calculate +DM and -DM
    high_diff = high_1d - np.roll(high_1d, 1)
    low_diff = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    # First values have no previous
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Calculate TR
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First bar
    
    # Calculate smoothed +DM, -DM, and TR using Wilder's smoothing (EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize first values
    tr_sum = np.zeros_like(tr_1d)
    plus_dm_sum = np.zeros_like(plus_dm)
    minus_dm_sum = np.zeros_like(minus_dm)
    
    tr_sum[period-1] = np.nansum(tr_1d[:period])  # Initial sum
    plus_dm_sum[period-1] = np.nansum(plus_dm[:period])
    minus_dm_sum[period-1] = np.nansum(minus_dm[:period])
    
    # Wilder's smoothing: today's value = yesterday's value * (1 - 1/period) + today's value * (1/period)
    for i in range(period, len(tr_1d)):
        tr_sum[i] = tr_sum[i-1] * (1 - alpha) + tr_1d[i] * alpha
        plus_dm_sum[i] = plus_dm_sum[i-1] * (1 - alpha) + plus_dm[i] * alpha
        minus_dm_sum[i] = minus_dm_sum[i-1] * (1 - alpha) + minus_dm[i] * alpha
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    # Handle division by zero
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    # Smooth DX to get ADX
    adx = np.zeros_like(dx)
    adx[period-1:] = dx[period-1:]  # Initial ADX value is DX
    for i in range(period, len(dx)):
        adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha
    
    # Align 1d ADX to 4h timeframe (wait for 1d bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3 AND 1d ADX > 25 (strong trend) AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S3 AND 1d ADX > 25 (strong trend) AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and adx_aligned[i] > 25 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals