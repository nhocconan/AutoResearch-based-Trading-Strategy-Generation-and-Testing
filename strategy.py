#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume spike confirmation, and choppiness regime filter.
# Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed to capture medium-term breakouts
# in both bull and bear markets by combining price structure (Donchian), trend filter (1d EMA34),
# volume confirmation, and regime filter (choppiness index) to avoid whipsaws in ranging conditions.
# Targets 75-200 total trades over 4 years.

name = "4h_Donchian20_Breakout_1dEMA34_VolumeChop_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use prior bar's levels (no look-ahead)
    highest_20 = np.roll(highest_20, 1)
    lowest_20 = np.roll(lowest_20, 1)
    highest_20[0] = np.nan
    lowest_20[0] = np.nan
    
    # Volume spike: > 1.8x 20-period average (balanced threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    # Choppiness Index (14-period) - regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / np.log10(14)) / np.log10(chop_denom)
    chop = np.where(chop_denom == 0, 50.0, chop)  # fallback when range is zero
    chop = np.nan_to_num(chop, nan=50.0)  # replace NaN with neutral value
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND close > 1d EMA34 (bullish trend)
            # AND volume spike AND choppiness < 61.8 (not too choppy)
            if (close[i] > highest_20[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i] and 
                chop[i] < 61.8):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower band AND close < 1d EMA34 (bearish trend)
            # AND volume spike AND choppiness < 61.8 (not too choppy)
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i] and 
                  chop[i] < 61.8):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian lower band OR 1d EMA34 (trend change)
            if close[i] < lowest_20[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian upper band OR 1d EMA34 (trend change)
            if close[i] > highest_20[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals