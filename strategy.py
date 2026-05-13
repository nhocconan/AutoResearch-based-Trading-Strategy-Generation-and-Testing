#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d ADX trend filter and volume spike confirmation.
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short).
# Only trade in direction of 1d ADX(14) > 25 (trending market).
# Requires volume > 2.0x 20-period average for confirmation.
# Exit when Williams %R reverts to -50 (mean reversion) or ADX < 20 (trend ends).
# Uses 6h timeframe for lower frequency, Williams %R for mean reversion in trends,
# 1d ADX for regime filter, volume spike for conviction. Works in bull (fade rallies) and bear (fade drops).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dADX_Volume_v1"
timeframe = "6h"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R(14) on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_6h = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close_6h) / (highest_high_6h - lowest_low_6h)
    
    # Volume filter: current 6h volume > 2.0x 20-period average
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_filter_6h = volume_6h > (2.0 * vol_ma_6h)
    
    # Get 1d data for ADX(14) trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    atr_smooth = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # +DI = 100 * smoothed +DM / smoothed TR
    # -DI = 100 * smoothed -DM / smoothed TR
    plus_di = 100 * plus_dm_smooth / atr_smooth
    minus_di = 100 * minus_dm_smooth / atr_smooth
    
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    # ADX = smoothed DX
    adx_1d = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_ma_6h[i]) or np.isnan(plus_di_1d_aligned[i]) or
            np.isnan(minus_di_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND ADX > 25 (trending) AND +DI > -DI (bullish trend) AND volume spike
            if williams_r_6h[i] < -80 and adx_1d_aligned[i] > 25 and plus_di_1d_aligned[i] > minus_di_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND ADX > 25 (trending) AND -DI > +DI (bearish trend) AND volume spike
            elif williams_r_6h[i] > -20 and adx_1d_aligned[i] > 25 and minus_di_1d_aligned[i] > plus_di_1d_aligned[i] and volume_filter_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R >= -50 (mean reversion) OR ADX < 20 (trend ending) OR -DI > +DI (trend reversal)
            if williams_r_6h[i] >= -50 or adx_1d_aligned[i] < 20 or minus_di_1d_aligned[i] > plus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R <= -50 (mean reversion) OR ADX < 20 (trend ending) OR +DI > -DI (trend reversal)
            if williams_r_6h[i] <= -50 or adx_1d_aligned[i] < 20 or plus_di_1d_aligned[i] > minus_di_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals