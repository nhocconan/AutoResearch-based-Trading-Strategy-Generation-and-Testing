#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter, volume spike, and ADX regime filter.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and ADX.
- Camarilla pivot levels: H3 (resistance 3) and L3 (support 3) from prior 1d OHLC.
- Breakout: Close > H3 (long) or Close < L3 (short) with volume > 2.0x 20-period volume MA.
- Trend filter: Only trade breakouts in direction of 1d EMA34 (long if close > EMA34, short if close < EMA34).
- Regime filter: Only trade when 1d ADX > 25 (trending market) to avoid choppy conditions.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Get 1d data for Camarilla pivots, EMA trend, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate prior 1d OHLC for Camarilla levels (H3, L3)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    h3 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low'])
    l3 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low'])
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d ADX for regime filter (trending market when ADX > 25)
    # ADX calculation: +DI, -DI, DX, then ADX = smoothed DX
    period = 14
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']) - pd.Series(df_1d['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    tr_smooth = pd.Series(atr).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    adx_values = adx.values
    
    # Align 1d indicators to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14*2)  # EMA34 + volume MA + ADX warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            if is_trending and volume_spike[i]:
                # Long breakout: close > H3 and close > 1d EMA34 (uptrend)
                if close[i] > h3_aligned[i] and close[i] > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < L3 and close < 1d EMA34 (downtrend)
                elif close[i] < l3_aligned[i] and close[i] < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Camarilla levels or opposite signal
            if close[i] < l3_aligned[i]:  # Exit when price falls below L3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Camarilla levels or opposite signal
            if close[i] > h3_aligned[i]:  # Exit when price rises above H3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_ADX_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0