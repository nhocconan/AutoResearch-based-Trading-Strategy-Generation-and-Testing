#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA200 trend filter and volume spike confirmation.
Long when Williams %R < -80 (oversold) AND price > 1d EMA200 (uptrend) AND volume > 2x 20-period average.
Short when Williams %R > -20 (overbought) AND price < 1d EMA200 (downtrend) AND volume > 2x 20-period average.
Exit when Williams %R reverts to -50 (mean reversion) or volume drops below average.
Uses 6h for Williams %R calculation and 1d for EMA200 trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures mean reversion in extremes,
volume spike confirms institutional interest, EMA200 filter ensures trend alignment.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
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
    
    # Highest high and lowest low over 14 periods
    hh = high_6h_series.rolling(window=14, min_periods=14).max().values
    ll = low_6h_series.rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (highest high - close) / (highest high - lowest low) * -100
    williams_r = ((hh - close_6h) / (hh - ll)) * -100
    # Handle division by zero (when hh == ll)
    williams_r = np.where((hh - ll) == 0, -50, williams_r)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    close_1d_series = pd.Series(close_1d)
    ema_200 = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 6h Williams %R to 6h timeframe (no alignment needed)
    williams_r_aligned = williams_r
    
    # Align 1d EMA200 to 6h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume average (20-period) on 6h
    volume_ma = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        ema = ema_200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND price > EMA200 (uptrend) AND volume > 2x avg
            if wr < -80 and price > ema and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < EMA200 (downtrend) AND volume > 2x avg
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

name = "6h_WilliamsR_1dEMA200_VolumeSpike"
timeframe = "6h"
leverage = 1.0