#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1
Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (EMA50), volume spike (>1.5x 20-bar avg volume), and ADX regime filter (ADX>25 for trend confirmation). Long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend. Uses discrete position sizing (0.25) to minimize fee drag. Targets 20-50 trades/year by requiring confluence of breakout, volume, trend, and regime. Works in bull/bear via trend filter: only takes longs in uptrend, shorts in downtrend, avoiding counter-trend whipsaws.
"""

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
    
    # Calculate ATR for stoploss (using 10-period ATR)
    atr_period = 10
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Calculate 1d EMA50 for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate ADX for regime filtering (trend strength)
    adx_period = 14
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    tr_14 = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    plus_di_14 = 100 * (pd.Series(plus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    minus_di_14 = 100 * (pd.Series(minus_dm).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=adx_period, min_periods=adx_period, adjust=False).mean().values
    
    # Calculate 20-bar average volume for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA, 14 for ADX)
    start_idx = max(50, 20, adx_period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(htf_trend[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Calculate Camarilla levels for today (using previous day's OHLC)
        # For 4h bars, we use daily OHLC from 1d data
        # Get previous day's close, high, low from 1d data
        prev_day_idx = i // 6  # Approximate: 6x 4h bars per day
        if prev_day_idx < 1 or prev_day_idx >= len(df_1d):
            # Hold current position if insufficient 1d data
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        prev_close = df_1d['close'].iloc[prev_day_idx - 1]
        prev_high = df_1d['high'].iloc[prev_day_idx - 1]
        prev_low = df_1d['low'].iloc[prev_day_idx - 1]
        
        # Camarilla R1, S1 levels
        R1 = close + (1.1/12) * (prev_high - prev_low)
        S1 = close - (1.1/12) * (prev_high - prev_low)
        
        # Volume spike condition: current volume > 1.5x 20-bar average
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Regime filter: only trade in trending markets (ADX > 25)
        strong_trend = adx[i] > 25
        
        # Entry logic: breakout with volume and trend confirmation
        if strong_trend and volume_spike:
            # Long: price breaks above R1 in uptrend
            if close[i] > R1 and htf_trend[i] == 1:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Short: price breaks below S1 in downtrend
            elif close[i] < S1 and htf_trend[i] == -1:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            else:
                # No breakout - hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Not ideal conditions - hold current position or exit if reversal signals
            if position == 1 and close[i] < S1:  # Exit long if price breaks S1
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > R1:  # Exit short if price breaks R1
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0