#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2
# Hypothesis: Refined version of Camarilla R1/S1 breakout with stricter volume confirmation (3x average) and ADX filter to reduce trades and avoid false breakouts.
# Uses ADX(14) > 25 to ensure trending market, reducing whipsaws in ranging conditions.
# Maintains 20-35 trades/year target to minimize fee drag while capturing strong institutional moves.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
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

    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla pivot levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)

    # 1-day EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 3.0 x 20-period average (stricter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma)

    # ADX filter for trending markets (14-period)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[np.nan], high[1:] - high[:-1]])
    down_move = np.concatenate([[np.nan], low[:-1] - low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values
    
    # DI values
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # ADX > 25 indicates trending market
    adx_filter = adx > 25

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(adx_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 with volume spike, above daily EMA34, and in trending market
            if (close[i] > r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i] and 
                adx_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike, below daily EMA34, and in trending market
            elif (close[i] < s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  adx_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or closes below daily EMA34
            if close[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or closes above daily EMA34
            if close[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals