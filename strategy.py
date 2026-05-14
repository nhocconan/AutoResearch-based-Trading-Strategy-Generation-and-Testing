#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d ATR volume spike + 1w EMA trend filter.
# Uses Williams Alligator (jaw/teeth/lips) on 12h to define trend direction and avoid ranging markets.
# Enters long when price > lips AND volume spike (ATR-normalized) AND 1w EMA > prior 1w EMA (bullish week).
# Enters short when price < lips AND volume spike AND 1w EMA < prior 1w EMA (bearish week).
# Exits on opposite lip touch or trend reversal. Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Targets 15-30 trades/year per symbol with discrete sizing (0.0, ±0.25) to minimize fee drag.

name = "12h_WilliamsAlligator_1dATRVolumeSpike_1wEMATrend_v1"
timeframe = "12h"
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
    
    # --- 12h Indicators (LTF) ---
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price = (high + low) / 2.0
    
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # ATR(14) for volatility normalization
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - close_shift), np.abs(low - close_shift)))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR-scaled volume MA: 20-period average of volume / ATR
    vol_atr_ratio = volume / (atr_14 + 1e-10)
    vol_atr_ma_20 = pd.Series(vol_atr_ratio).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_atr_ratio > (2.0 * vol_atr_ma_20)  # stricter threshold to reduce trades
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR-based volume spike on 1d (for confirmation)
    high_shift_1d = np.roll(high_1d, 1)
    low_shift_1d = np.roll(low_1d, 1)
    close_shift_1d = np.roll(close_1d, 1)
    high_shift_1d[0] = high_1d[0]
    low_shift_1d[0] = low_1d[0]
    close_shift_1d[0] = close_1d[0]
    
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - close_shift_1d), np.abs(low_1d - close_shift_1d)))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_atr_ratio_1d = volume / (atr_14_1d + 1e-10)  # volume from 12h data, normalized by 1d ATR
    vol_atr_ma_20_1d = pd.Series(vol_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_atr_ratio_1d > (2.0 * vol_atr_ma_20_1d)
    
    # Align 1d volume spike to 12h
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 34-period EMA on 1w for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Prior week EMA for trend direction (bullish if current > prior)
    ema_34_1w_prior = np.roll(ema_34_1w_aligned, 1)
    ema_34_1w_prior[0] = ema_34_1w_aligned[0]
    ema_34_1w_rising = ema_34_1w_aligned > ema_34_1w_prior
    ema_34_1w_falling = ema_34_1w_aligned < ema_34_1w_prior
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # warmup for Alligator and HTF alignment
        # Skip if missing data
        if (np.isnan(lips[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleep condition: jaw, teeth, lips intertwined (ranging market)
        # Lips between jaw and teeth OR teeth between jaw and lips
        lips_between = (jaw[i] <= lips[i] <= teeth[i]) or (teeth[i] <= lips[i] <= jaw[i])
        teeth_between = (jaw[i] <= teeth[i] <= lips[i]) or (lips[i] <= teeth[i] <= jaw[i])
        alligator_sleep = lips_between or teeth_between
        
        if position == 0:
            # LONG: Price > lips, volume spike, Alligator awake (lips > teeth > jaw), weekly EMA rising
            if (close[i] > lips[i] and
                volume_spike_1d_aligned[i] and
                lips[i] > teeth[i] and
                teeth[i] > jaw[i] and
                ema_34_1w_rising[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < lips, volume spike, Alligator awake (lips < teeth < jaw), weekly EMA falling
            elif (close[i] < lips[i] and
                  volume_spike_1d_aligned[i] and
                  lips[i] < teeth[i] and
                  teeth[i] < jaw[i] and
                  ema_34_1w_falling[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < lips OR Alligator starts sleeping (lips < teeth) OR weekly EMA turns falling
            if (close[i] < lips[i] or
                lips[i] < teeth[i] or
                not ema_34_1w_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > lips OR Alligator starts sleeping (lips > teeth) OR weekly EMA turns rising
            if (close[i] > lips[i] or
                lips[i] > teeth[i] or
                not ema_34_1w_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals