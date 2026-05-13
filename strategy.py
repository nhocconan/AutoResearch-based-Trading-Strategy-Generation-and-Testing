#!/usr/bin/env python3
# Hypothesis: 1d Camarilla pivot (R3/S3) breakout with 1w EMA50 trend filter, volume confirmation (>1.5x 20-bar avg volume), and choppiness regime filter (CHOP < 38.2 = trending). 
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# Camarilla pivots identify key support/resistance levels; 1w EMA50 ensures higher timeframe trend alignment; 
# Volume confirmation filters low-participation breakouts; Chop filter ensures we only trade in trending markets.
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "1d_Camarilla_R3_S3_Breakout_1wEMA50_Volume_Chop_Filter"
timeframe = "1d"
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla pivot levels (R3, S3) from prior day only
    lookback_cam = 1
    prior_close = pd.Series(close).shift(1).values
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 4
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 4
    
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
    
    start_idx = max(1, lookback_vol, lookback_chop)  # Start after sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i]) or
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
            # LONG: Price breaks above Camarilla R3, close > 1w EMA50, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1w EMA50, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals