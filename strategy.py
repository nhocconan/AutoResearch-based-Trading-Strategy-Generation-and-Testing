#!/usr/bin/env python3
# 6h_adx_di_volume_v1
# Hypothesis: 6h strategy using ADX(14) and +DI/-DI crossover for trend strength/direction, filtered by 1d EMA200 for higher timeframe bias, and volume > 1.5x 20-period average for confirmation.
# ADX > 25 ensures we only trade in trending markets, reducing whipsaw in ranging conditions.
# Works in both bull and bear markets by going long when +DI > -DI and short when -DI > +DI, aligned with 1d trend.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_di_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # ADX and DI calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    plus_dm_smooth = np.zeros_like(plus_dm)
    minus_dm_smooth = np.zeros_like(minus_dm)
    
    # Initialize first value with simple average
    atr[period] = np.nansum(tr[1:period+1]) / period
    plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1]) / period
    minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1]) / period
    
    # Wilder smoothing
    for i in range(period+1, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, (plus_dm_smooth / atr) * 100, 0)
    minus_di = np.where(atr != 0, (minus_dm_smooth / atr) * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    
    # ADX: smoothed DX
    adx = np.zeros_like(dx)
    adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
    for i in range(2*period+1, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX < 20 (trend weakening) OR -DI > +DI (trend reversal) OR price below EMA200
            if adx_aligned[i] < 20 or minus_di_aligned[i] > plus_di_aligned[i] or close[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX < 20 (trend weakening) OR +DI > -DI (trend reversal) OR price above EMA200
            if adx_aligned[i] < 20 or plus_di_aligned[i] > minus_di_aligned[i] or close[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and ADX > 25 (strong trend)
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            strong_trend = adx_aligned[i] > 25
            
            if volume_confirmed and strong_trend:
                # Long: +DI > -DI and price above EMA200 (bullish alignment)
                if plus_di_aligned[i] > minus_di_aligned[i] and close[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: -DI > +DI and price below EMA200 (bearish alignment)
                elif minus_di_aligned[i] > plus_di_aligned[i] and close[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals