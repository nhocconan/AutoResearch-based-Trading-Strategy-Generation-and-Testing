#!/usr/bin/env python3
# 1h_TRIX_Volume_Regime
# Hypothesis: TRIX (triple EMA) captures momentum with reduced noise. Combine with volume confirmation and chop regime filter to avoid false signals. Use 1d EMA50 for trend filter and 4h TRIX for signal direction. Trade only during 08-20 UTC session to avoid low-liquidity hours. Target: 20-40 trades/year to minimize fee drag. Works in bull/bear markets by following 1d trend direction.

name = "1h_TRIX_Volume_Regime"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4h data for TRIX calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on 4h close: triple EMA then percent change
    ema1 = pd.Series(df_4h['close']).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # TRIX as percentage
    trix_values = trix.values
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix_values)
    
    # Calculate 4h Choppiness Index for regime filter
    atr_period = 14
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    max_high = df_4h['high'].rolling(window=atr_period, min_periods=atr_period).max()
    min_low = df_4h['low'].rolling(window=atr_period, min_periods=atr_period).min()
    chop = 100 * np.log10(atr.rolling(window=atr_period, min_periods=atr_period).sum() / (max_high - min_low)) / np.log10(atr_period)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop_values)
    
    # Volume confirmation on 1h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1d EMA50 (50), 4h TRIX (12*3=36), volume MA (20), chop (14*2=28)
    start_idx = max(50, 36, 20, 28)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(trix_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        h = hours[i]
        in_session = (8 <= h <= 20)
        
        # 1d trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Chop regime: < 38.2 = trending, > 61.8 = ranging
        trending_regime = chop_aligned[i] < 38.2
        
        if position == 0:
            if in_session and volume_confirm and trending_regime:
                # Long entry: uptrend + TRIX turning up (positive and rising)
                if uptrend and trix_aligned[i] > 0 and trix_aligned[i] > trix_aligned[i-1]:
                    signals[i] = 0.20
                    position = 1
                # Short entry: downtrend + TRIX turning down (negative and falling)
                elif downtrend and trix_aligned[i] < 0 and trix_aligned[i] < trix_aligned[i-1]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: trend breaks or TRIX turns down
            if not uptrend or trix_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or TRIX turns up
            if not downtrend or trix_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals