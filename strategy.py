#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume confirmation (>1.8x 20-bar avg volume), and choppiness regime filter (CHOP < 38.2 = trending). 
# Targets 75-150 trades over 4 years on 4h timeframe. Camarilla levels provide institutional support/resistance; 
# 1d EMA50 ensures higher timeframe trend alignment; volume filters low-conviction breakouts; 
# Chop filter avoids ranging markets. Discrete sizing 0.25 minimizes fee drag while maintaining edge.

name = "4h_Camarilla_R3S3_Breakout_1dEMA50_Volume_Chop_v2"
timeframe = "4h"
leverage = 1.0

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
    open_price = prices['open'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior day (using 1d OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 (inner levels) for breakouts
    prior_day_close = df_1d['close'].shift(1).values
    prior_day_high = df_1d['high'].shift(1).values
    prior_day_low = df_1d['low'].shift(1).values
    prior_day_range = prior_day_high - prior_day_low
    
    camarilla_r3 = prior_day_close + 1.1 * prior_day_range
    camarilla_s3 = prior_day_close - 1.1 * prior_day_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    # Calculate Choppiness Index (14-period) for regime filter
    lookback_chop = 14
    tr1 = pd.Series(high).rolling(lookback_chop).max() - pd.Series(low).rolling(lookback_chop).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback_chop, min_periods=lookback_chop).sum()
    sum_close_diff = abs(pd.Series(close) - pd.Series(close).shift(1)).rolling(window=lookback_chop, min_periods=lookback_chop).sum()
    chop = 100 * np.log10(sum_close_diff / atr) / np.log10(lookback_chop)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    min_start = max(lookback_vol, lookback_chop, 1)
    for i in range(min_start, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness regime filter: only trade in trending markets (CHOP < 38.2)
        if chop_values[i] >= 38.2:
            # In ranging market, force flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA50, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA50, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals