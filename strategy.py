#!/usr/bin/env python3
name = "4h_Donchian20_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian(20) on 4h timeframe
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14, 4)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > high_max[i-1] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with volume and daily downtrend
            elif close[i] < low_min[i-1] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Stoploss: price drops below entry - 2*ATR
            # Since we don't track entry price, use trailing stop: highest high since entry - 2*ATR
            # Simplified: exit if price drops below Donchian lower band or volume drops
            if close[i] < low_min[i] or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Stoploss: price rises above entry + 2*ATR
            # Simplified: exit if price rises above Donchian upper band or volume drops
            if close[i] > high_max[i] or volume[i] < vol_ma_4[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# - Donchian(20) breakout captures momentum in both bull and bear markets
# - Daily EMA(34) trend filter ensures we only trade in direction of higher timeframe trend
# - Volume confirmation (1.5x average) filters out false breakouts
# - Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend)
# - Exit when price returns to Donchian channel or volume drops significantly
# - Position size 0.25 limits drawdown during 2022-like crashes (~17% loss vs 77% for 100%)
# - Target: 20-50 trades per year to avoid fee drag while capturing strong moves
# - Uses tight entry conditions to stay under 400 total 4h trades over 4 years
# - Combines proven elements: price channel breakout + trend filter + volume confirmation