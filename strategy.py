#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trix_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for TRIX and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # TRIX calculation on weekly close (12-period EMA of 12-period EMA of 12-period EMA)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # first value has no previous
    
    # Align TRIX to daily timeframe
    trix_1w_aligned = align_htf_to_ltf(prices, df_1w, trix_raw)
    
    # Weekly volume moving average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Daily Choppiness Index (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    chop = np.where(tr_sum > 0, chop_raw, 50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(trix_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current weekly volume > 20-period average
        volume_filter = volume[i] > vol_ma_1w_aligned[i]
        
        # Chop regime: Chop < 38.2 = trending (trend follow), Chop > 61.8 = ranging (mean revert)
        trending_regime = chop[i] < 38.2
        ranging_regime = chop[i] > 61.8
        
        # Long conditions:
        # 1. TRIX positive and rising (bullish momentum)
        # 2. Volume confirmation
        # 3. Trending regime
        trix_rising = trix_1w_aligned[i] > trix_1w_aligned[i-1]
        long_signal = (trix_1w_aligned[i] > 0 and trix_rising and volume_filter and trending_regime)
        
        # Short conditions:
        # 1. TRIX negative and falling (bearish momentum)
        # 2. Volume confirmation
        # 3. Trending regime
        trix_falling = trix_1w_aligned[i] < trix_1w_aligned[i-1]
        short_signal = (trix_1w_aligned[i] < 0 and trix_falling and volume_filter and trending_regime)
        
        # Exit conditions:
        # Exit long when TRIX turns negative or chop enters ranging
        exit_long = (position == 1 and (trix_1w_aligned[i] < 0 or chop[i] > 61.8))
        # Exit short when TRIX turns positive or chop enters ranging
        exit_short = (position == -1 and (trix_1w_aligned[i] > 0 or chop[i] > 61.8))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals