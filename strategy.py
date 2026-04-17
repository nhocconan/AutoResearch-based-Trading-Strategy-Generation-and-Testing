#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme reversal with 1d EMA200 filter and volume spike confirmation.
Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA200 (bullish bias) AND volume > 2x average.
Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA200 (bearish bias) AND volume > 2x average.
Exit when Williams %R returns to -50 (mean reversion) OR volume drops below average.
Uses 6h for Williams %R calculation and 1d for EMA filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures reversals in bear market rallies and bull market pullbacks.
Volume spike confirms institutional interest. EMA200 filter ensures trend alignment.
Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
"""

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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R on 6h timeframe (14-period)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    close_6h_series = pd.Series(close_6h)
    
    highest_high = high_6h_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_6h_series.rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Get 1d data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema200 = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema = ema200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > EMA200 AND volume > 2x average
            if wr < -80 and price > ema and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < EMA200 AND volume > 2x average
            elif wr > -20 and price < ema and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (mean reversion) OR volume < average
            if wr > -50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (mean reversion) OR volume < average
            if wr < -50 or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_EMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0