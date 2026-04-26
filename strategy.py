#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation (2.0x), and chop regime filter (CHOP > 61.8 = range, < 38.2 = trend). 
In ranging markets (CHOP > 61.8): fade breaks (short at R3, long at S3). 
In trending markets (CHOP < 38.2): follow breaks (long at R3, short at S3). 
Uses 1d trend alignment for higher-timeframe bias. Discrete position sizing (0.25) minimizes fee churn. 
Designed for 4h timeframe targeting 75-200 total trades over 4 years (19-50/year). Works in bull/bear via regime adaptation and 1d trend filter.
"""

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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels from 1d OHLC (using previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume filter: volume > 2.0 * volume_ma(20) for tight confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Choppiness Index regime filter (14-period)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_14[:14] = np.nan  # first 14 values invalid due to roll
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14 / np.log(max_high_14 - min_low_14)) / np.log10(14)
    chop_regime = np.where(chop > 61.8, 1, np.where(chop < 38.2, -1, 0))  # 1=range, -1=trend, 0=neutral
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14 for chop)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(trend_1d[i]) or np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime-dependent logic
        if chop_regime[i] == 1:  # Ranging market: fade breaks
            if position == 0:
                # Short at R3, long at S3 (fade the break)
                if close[i] > camarilla_r3_aligned[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] < camarilla_s3_aligned[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Hold long
                signals[i] = 0.25
                # Exit: Price rises above Camarilla R3 (stop fading long) OR 1d trend turns up
                if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                # Hold short
                signals[i] = -0.25
                # Exit: Price falls below Camarilla S3 (stop fading short) OR 1d trend turns down
                if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                    signals[i] = 0.0
                    position = 0
        elif chop_regime[i] == -1:  # Trending market: follow breaks
            if position == 0:
                # Long: Price breaks above Camarilla R3 AND 1d uptrend AND volume spike
                if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Price breaks below Camarilla S3 AND 1d downtrend AND volume spike
                elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # Hold long
                signals[i] = 0.25
                # Exit: Price falls below Camarilla S3 OR 1d trend turns down
                if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                # Hold short
                signals[i] = -0.25
                # Exit: Price rises above Camarilla R3 OR 1d trend turns up
                if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                    signals[i] = 0.0
                    position = 0
        else:  # Neutral regime: no position
            signals[i] = 0.0
            position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "4h"
leverage = 1.0