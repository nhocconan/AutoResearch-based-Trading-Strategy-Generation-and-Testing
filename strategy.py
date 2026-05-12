#!/usr/bin/env python3
# 4h_Keltner_Breakout_1dTrend_VolumeFilter
# Hypothesis: On 4h timeframe, enter long when price breaks above Keltner upper band with price > daily EMA200 and volume spike (>1.5x 20-period MA).
# Enter short when price breaks below Keltner lower band with price < daily EMA200 and volume spike.
# Exit when price crosses back inside Keltner bands.
# Uses daily timeframe for trend filter to capture longer-term trend and avoid counter-trend trades.
# Designed to work in both bull and bear markets by following the daily trend while using volatility-based bands for entry timing.
# Targets 25-40 trades/year for low fee drag.

name = "4h_Keltner_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Keltner Channel (20, 10) on 4h data
    keltner_period = 20
    keltner_mult = 10.0  # Using ATR multiplier of 10 for wider bands
    
    # EMA of typical price for middle band
    typical_price = (high + low + close) / 3.0
    ema_tp = pd.Series(typical_price).ewm(span=keltner_period, adjust=False, min_periods=keltner_period).mean().values
    
    # Average True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=keltner_period, adjust=False, min_periods=keltner_period).mean().values
    
    upper_keltner = ema_tp + (keltner_mult * atr)
    lower_keltner = ema_tp - (keltner_mult * atr)
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema200 = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume confirmation: 20-period moving average on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily EMA200 to 4h timeframe
    daily_ema200_aligned = align_htf_to_ltf(prices, df_1d, daily_ema200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(daily_ema200_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        upper_keltner_val = upper_keltner[i]
        lower_keltner_val = lower_keltner[i]
        daily_trend = daily_ema200_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above upper Keltner band with price > daily EMA200 and volume > 1.5x MA
            if close[i] > upper_keltner_val and close[i] > daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner band with price < daily EMA200 and volume > 1.5x MA
            elif close[i] < lower_keltner_val and close[i] < daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back inside Keltner bands (below upper band)
            if close[i] < upper_keltner_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back inside Keltner bands (above lower band)
            if close[i] > lower_keltner_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals