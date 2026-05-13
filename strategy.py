#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and ADX trend filter.
# Long when price breaks above Camarilla R1 (1d) AND volume > 2.0x 20-period average AND ADX(14) > 25 (trending)
# Short when price breaks below Camarilla S1 (1d) AND volume > 2.0x 20-period average AND ADX(14) > 25
# Exit when price reverts to Camarilla pivot point (PP) or ADX < 20 (range)
# Uses Camarilla pivot levels from 1d for structure, volume spike for confirmation, ADX for regime filter.
# Target: 75-150 total trades over 4 years (19-37/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Camarilla_R1S1_Breakout_1dVolume_ADX_v1"
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
    
    # Get 1d data for Camarilla pivot levels, volume, and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current 1d volume > 2.0x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # ADX calculation for trend strength
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DM_smooth = smoothed +DM (Wilder's smoothing)
    # -DM_smooth = smoothed -DM
    # TR_smooth = smoothed TR
    # +DI = 100 * +DM_smooth / TR_smooth
    # -DI = 100 * -DM_smooth / TR_smooth
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed DX
    
    # Calculate +DM, -DM, TR
    high_shift = np.roll(high_1d, 1)
    low_shift = np.roll(low_1d, 1)
    close_shift = np.roll(close_1d, 1)
    high_shift[0] = np.nan
    low_shift[0] = np.nan
    close_shift[0] = np.nan
    
    plus_dm = np.where((high_1d - high_shift) > (low_shift - low_1d), np.maximum(high_1d - high_shift, 0), 0)
    minus_dm = np.where((low_shift - low_1d) > (high_1d - high_shift), np.maximum(low_shift - low_1d, 0), 0)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_shift)
    tr3 = np.abs(low_1d - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    # Initialize smoothed values
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    tr_smooth = np.zeros_like(tr)
    
    # First value is simple average
    plus_dm_smooth[period-1] = np.nansum(plus_dm[:period])
    minus_dm_smooth[period-1] = np.nansum(minus_dm[:period])
    tr_smooth[period-1] = np.nansum(tr[:period])
    
    # Subsequent values: smoothed = previous_smooth - (previous_smooth / period) + current_value
    for i in range(period, len(plus_dm)):
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1] / period) + tr[i]
    
    # Calculate +DI, -DI, DX, ADX
    plus_di = 100.0 * plus_dm_smooth / tr_smooth
    minus_di = 100.0 * minus_dm_smooth / tr_smooth
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX is smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period-1] = np.nansum(dx[period:2*period])  # First ADX value
    for i in range(2*period, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(pp_1d_aligned[i]) or np.isnan(volume_filter_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 AND volume confirmation AND ADX > 25 (trending)
            if close[i] > r1_1d_aligned[i] and volume_filter_1d_aligned[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 AND volume confirmation AND ADX > 25 (trending)
            elif close[i] < s1_1d_aligned[i] and volume_filter_1d_aligned[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to PP OR ADX < 20 (range)
            if close[i] <= pp_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reverts to PP OR ADX < 20 (range)
            if close[i] >= pp_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals