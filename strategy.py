#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSqueeze"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Bollinger Bands for volatility squeeze filter (20, 2)
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 1d (OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values
    
    prev_high = high_1d[:-1]
    prev_low = low_1d[:-1]
    prev_close = close_1d[:-1]
    
    hl_range = prev_high - prev_low
    r3_levels = prev_close + hl_range * 1.1 / 2
    s3_levels = prev_close - hl_range * 1.1 / 2
    
    r3_per_day = np.full(len(df_1d), np.nan)
    s3_per_day = np.full(len(df_1d), np.nan)
    r3_per_day[1:] = r3_levels
    s3_per_day[1:] = s3_levels
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_per_day)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_per_day)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for BB width MA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(bb_width_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility squeeze condition: current BB width < 80% of 20-period average
        volatility_squeeze = bb_width[i] < bb_width_ma[i] * 0.8
        
        if position == 0:
            # Long: price > R3, above EMA34, volume spike, volatility squeeze
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5 and
                volatility_squeeze):
                signals[i] = 0.25
                position = 1
            # Short: price < S3, below EMA34, volume spike, volatility squeeze
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5 and
                  volatility_squeeze):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price < S3 or below EMA34
            if (close[i] < s3_aligned[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price > R3 or above EMA34
            if (close[i] > r3_aligned[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation, and volatility squeeze.
# The volatility squeeze (Bollinger Band width contraction) identifies periods of low volatility that often precede breakouts.
# Combined with volume confirmation, this increases the probability of a genuine breakout rather than a false move.
# Works in bull markets (buy breakouts above R3 in uptrend) and bear markets (sell breakdowns below S3 in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~20-40 trades/year).