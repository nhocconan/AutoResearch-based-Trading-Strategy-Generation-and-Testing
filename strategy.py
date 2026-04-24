#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d/1w regime filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for Bull/Bear Power EMA trend, 1w for market regime (bull/bear/range).
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13). Measures buying/selling pressure.
- Regime filter: 1w ADX > 25 and +DI > -DI = bull regime (favor longs), ADX > 25 and -DI > +DI = bear regime (favor shorts), else range.
- Entry: In bull regime: Long when Bull Power > 0 and rising (Bull Power > prev) AND volume > 1.5 * 20 MA.
         In bear regime: Short when Bear Power < 0 and falling (Bear Power < prev) AND volume > 1.5 * 20 MA.
         In range: Mean reversion at Bollinger Bands (20,2) - Long at lower band reversal, Short at upper band reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid false signals.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull/bear via regime adaptation, and range via mean reversion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 (Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for ADX regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate EMA(13) on 1d close for Elder Ray
    ema_13 = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align 1d Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate ADX (14-period) on 1w for regime
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX and DI to 6h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
    
    # Bollinger Bands (20,2) on 6h for range regime mean reversion
    lookback = 20
    bb_mid = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    bb_std = pd.Series(close).rolling(window=lookback, min_periods=lookback).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine regime from 1w ADX and DI
        adx_val = adx_aligned[i]
        plus_di_val = plus_di_aligned[i]
        minus_di_val = minus_di_aligned[i]
        
        if adx_val > 25 and plus_di_val > minus_di_val:
            regime = 'bull'  # bull trend
        elif adx_val > 25 and minus_di_val > plus_di_val:
            regime = 'bear'  # bear trend
        else:
            regime = 'range'  # ranging market
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        prev_bull = bull_power_aligned[i-1] if i > 0 else 0
        prev_bear = bear_power_aligned[i-1] if i > 0 else 0
        
        if position == 0:
            # Check for entry signals based on regime
            if volume_spike[i]:
                if regime == 'bull':
                    # Long when Bull Power > 0 and rising (strong buying pressure)
                    if bull_power_aligned[i] > 0 and bull_power_aligned[i] > prev_bull:
                        signals[i] = 0.25
                        position = 1
                elif regime == 'bear':
                    # Short when Bear Power < 0 and falling (strong selling pressure)
                    if bear_power_aligned[i] < 0 and bear_power_aligned[i] < prev_bear:
                        signals[i] = -0.25
                        position = -1
                else:  # range regime
                    # Mean reversion at Bollinger Bands
                    # Long when price touches lower band and reverses up
                    if curr_low <= bb_lower[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper band and reverses down
                    elif curr_high >= bb_upper[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative OR regime shifts to bear
            if bull_power_aligned[i] <= 0 or regime == 'bear':
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive OR regime shifts to bull
            if bear_power_aligned[i] >= 0 or regime == 'bull':
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1d1wRegime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0