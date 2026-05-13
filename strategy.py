#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation (>1.8x 20-bar avg volume), and choppiness regime filter (CHOP < 38.2 = trending).
# Uses discrete sizing 0.28 to target 80-120 total trades over 4 years on 4h timeframe.
# Camarilla levels provide institutional support/resistance; 1d EMA34 ensures higher timeframe trend alignment;
# Volume confirmation filters low-participation breakouts; Chop filter ensures trending markets only.
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_Chop_v3"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (R3, S3) from prior day only
    lookback_cam = 24  # 24 * 1h = 1d, but we use prior completed 1d bar
    prior_close = pd.Series(close).shift(24).values  # prior 1d close
    prior_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(24).values
    prior_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(24).values
    
    # Camarilla R3 and S3 levels
    rang = prior_high - prior_low
    r3 = prior_close + rang * 1.1 / 4
    s3 = prior_close - rang * 1.1 / 4
    
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
    
    start_idx = max(24, lookback_vol, lookback_chop) + 1
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness regime filter: only trade in trending markets (CHOP < 38.2)
        if chop_values[i] >= 38.2:
            # In ranging or choppy market, force flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike
            if (high[i] > r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.28
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike
            elif (low[i] < s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.28
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < s3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals