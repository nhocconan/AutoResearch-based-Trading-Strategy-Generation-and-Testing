# 12h_Camarilla_R1_S1_Breakout_1dTrend_Filtered
# Hypothesis: On 12h timeframe, buy when price breaks above Camarilla R1 from previous 1d with volume >2x average and 1d EMA34 trending up; sell when price breaks below Camarilla S1 with volume >2x average and 1d EMA34 trending down. Added ADX(14) > 25 trend filter to reduce false breakouts in ranging markets. Targets 15-30 trades per year to reduce fee drag and improve generalization in bull/bear markets.
# Uses discrete position sizing (0.25) to minimize churn and respects all MTF data loading rules.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Filtered"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = close + (high - low) * 1.12/12, S1 = close - (high - low) * 1.12/12
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.12 / 12
    camarilla_s1 = close_1d - range_1d * 1.12 / 12

    # Use previous 1d bar's levels (shift by 1)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan

    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_prev)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # 1d ADX(14) for trend strength filter
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])  # first value is sum
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    # Calculate DI+ and DI-
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = smooth_wilder(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)

    # Volume confirmation: volume > 2x 24-period average (approx 12 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_24[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 1d uptrend + volume spike + ADX > 25
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 2.0 and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 1d downtrend + volume spike + ADX > 25
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 2.0 and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 OR trend turns down OR ADX weakens
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema34_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 OR trend turns up OR ADX weakens
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema34_1d_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals