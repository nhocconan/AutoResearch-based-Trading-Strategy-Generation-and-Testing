#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA200 trend filter and Bollinger Band squeeze confirmation.
# Long when Williams %R < -80 (oversold), price > 1d EMA200 (uptrend), and BB width < 20th percentile (low volatility squeeze).
# Short when Williams %R > -20 (overbought), price < 1d EMA200 (downtrend), and BB width < 20th percentile.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts (mean reversion completion).
# Uses discrete position sizing 0.25. Target: 50-150 total trades over 4 years on 6h timeframe.
# Williams %R identifies overextended moves, EMA200 filters for trend alignment, BB squeeze confirms low volatility environment for mean reversion.
# This combination avoids false signals in high volatility and works in both bull (buy dips) and bear (sell rallies) markets.

name = "6h_WilliamsR_MeanReversion_1dEMA200_BBSqueeze_v1"
timeframe = "6h"
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
    
    # Williams %R (14-period)
    lookback_wr = 14
    if n < lookback_wr:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=lookback_wr, min_periods=lookback_wr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_wr, min_periods=lookback_wr).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Bollinger Band width (20-period, 2 std dev) for squeeze detection
    lookback_bb = 20
    if n < lookback_bb:
        return np.zeros(n)
    
    sma_20 = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).mean().values
    std_20 = pd.Series(close).rolling(window=lookback_bb, min_periods=lookback_bb).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    # Percentile lookback for squeeze definition (50 periods ~ 150h/6.25d)
    lookback_percentile = 50
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=lookback_percentile, min_periods=lookback_percentile).quantile(0.20).values
    is_squeeze = bb_width < bb_width_percentile
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d EMA200 to 6h timeframe (wait for 1d bar to close)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_wr, lookback_bb, lookback_percentile) + 5  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(is_squeeze[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), uptrend (close > EMA200), and BB squeeze
            if (williams_r[i] < -80 and 
                close[i] > ema_200_1d_aligned[i] and 
                is_squeeze[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), downtrend (close < EMA200), and BB squeeze
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  is_squeeze[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (mean reversion)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (mean reversion)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals