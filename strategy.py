#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h ADX trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold exit) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Williams %R crosses below -80 (overbought exit) AND 12h ADX > 25 AND volume > 1.5x 20-period average.
# Exit when Williams %R crosses back to neutral zone (-50).
# This strategy captures mean reversion in trending markets, avoiding choppy conditions via ADX filter.
# Williams %R is effective in both bull and bear markets for identifying reversal points.
# Volume confirmation ensures institutional participation and reduces false signals.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.

name = "6h_WilliamsR_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14)
    wr_length = 14
    highest_high = pd.Series(high).rolling(window=wr_length, min_periods=wr_length).max().values
    lowest_low = pd.Series(low).rolling(window=wr_length, min_periods=wr_length).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 12h ADX (14) for trend strength
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # ADX > 25 (trending market)
    adx_trending = adx_12h_aligned > 25
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(wr_length, 30)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(adx_trending[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 from below (exiting oversold)
            long_cross = (williams_r[i] > -20) and (williams_r[i-1] <= -20)
            # Short entry: Williams %R crosses below -80 from above (exiting overbought)
            short_cross = (williams_r[i] < -80) and (williams_r[i-1] >= -80)
            
            if long_cross and adx_trending[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            elif short_cross and adx_trending[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (return to neutral)
            if williams_r[i] < -50 and williams_r[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (return to neutral)
            if williams_r[i] > -50 and williams_r[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals