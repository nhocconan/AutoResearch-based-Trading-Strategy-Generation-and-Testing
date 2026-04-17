#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + Volume Spike + 1d EMA200 Trend Filter.
Long when Williams %R < -90 (oversold) AND volume > 2.0x 20-period average AND close > 1d EMA200 (uptrend).
Short when Williams %R > -10 (overbought) AND volume > 2.0x 20-period average AND close < 1d EMA200 (downtrend).
Exit when Williams %R reverts to -50 (mean reversion) OR volume < 1.2x average (momentum fade).
Uses 6h for Williams %R and volume, 1d for EMA200 trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R extremes capture reversals in bear market rallies and bull market pullbacks. Volume confirmation ensures participation. EMA200 filter ensures alignment with higher timeframe trend.
Works in bull markets (buys oversold dips in uptrend) and bear markets (sells overbought rallies in downtrend).
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
    
    # Get 6h data for Williams %R and volume
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate Williams %R on 6h timeframe (14-period)
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / np.where((highest_high - lowest_low) != 0, (highest_high - lowest_low), np.inf)
    
    # Get 1d data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d timeframe
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # Align 6h Williams %R to 6h timeframe (no alignment needed)
    williams_r_aligned = williams_r
    
    # Align 1d EMA200 to 6h timeframe
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Align 6h volume MA to 6h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
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
        ema200 = ema200_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: WR < -90 (oversold) AND volume > 2.0x avg AND price > EMA200 (uptrend)
            if wr < -90 and vol > 2.0 * vol_ma and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: WR > -10 (overbought) AND volume > 2.0x avg AND price < EMA200 (downtrend)
            elif wr > -10 and vol > 2.0 * vol_ma and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WR > -50 (mean reversion) OR volume < 1.2x average (momentum fade)
            if wr > -50 or vol < 1.2 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WR < -50 (mean reversion) OR volume < 1.2x average (momentum fade)
            if wr < -50 or vol < 1.2 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_VolumeSpike_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0