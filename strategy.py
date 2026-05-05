#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h ADX trend filter and volume confirmation
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h ADX > 25 AND volume > 1.5x average
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h ADX > 25 AND volume > 1.5x average
# Exit when power fails OR ADX < 20 (trend weakens)
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Elder Ray measures bull/bear strength relative to EMA, ADX filters for trending markets only,
# volume confirms participation. Works in bull markets via strong Bull Power and bear markets via strong Bear Power.

name = "6h_ElderRay_BullBearPower_12hADX25_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    if len(close) >= 13:
        ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    else:
        return np.zeros(n)
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # Get 12h data ONCE before loop for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original indices
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    if len(tr) >= tr_period:
        tr_sum = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).sum().values
        dm_plus_sum = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).sum().values
        dm_minus_sum = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).sum().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_sum / tr_sum
        di_minus = 100 * dm_minus_sum / tr_sum
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    else:
        adx = np.full(len(close_12h), np.nan)
    
    # ADX trend filter: >25 = strong trend, <20 = weak trend
    strong_trend = adx > 25
    weak_trend = adx < 20
    
    # Align 12h indicators to 6h timeframe
    strong_trend_aligned = align_htf_to_ltf(prices, df_12h, strong_trend.astype(float))
    weak_trend_aligned = align_htf_to_ltf(prices, df_12h, weak_trend.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Power momentum: rising/falling power
    bull_power_rising = bull_power > np.concatenate([[np.nan], bull_power[:-1]])
    bear_power_rising = bear_power > np.concatenate([[np.nan], bear_power[:-1]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(strong_trend_aligned[i]) or 
            np.isnan(weak_trend_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            np.isnan(bull_power_rising[i]) or
            np.isnan(bear_power_rising[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power positive AND rising AND strong trend AND volume spike
            if (bull_power[i] > 0 and 
                bull_power_rising[i] and 
                strong_trend_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power positive AND rising AND strong trend AND volume spike
            elif (bear_power[i] > 0 and 
                  bear_power_rising[i] and 
                  strong_trend_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power fails OR weak trend
            if (bull_power[i] <= 0 or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power fails OR weak trend
            if (bear_power[i] <= 0 or 
                weak_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals