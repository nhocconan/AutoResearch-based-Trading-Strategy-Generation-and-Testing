#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d Volume Spike and 1w EMA200 Trend Filter.
Long when Williams %R < -80 (oversold) AND 1d volume > 2.0x 20-day average AND price > 1w EMA200 (bullish bias).
Short when Williams %R > -20 (overbought) AND 1d volume > 2.0x 20-day average AND price < 1w EMA200 (bearish bias).
Exit when Williams %R returns to -50 (mean reversion) OR volume spike ends.
Uses 6h for Williams %R timing, 1d for volume confirmation, 1w for trend filter to avoid counter-trend trades.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R catches reversals in bear market rallies,
volume confirmation ensures participation, weekly EMA filter avoids fighting the major trend.
Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Williams %R on 6h timeframe (14-period)
    high_6h_series = pd.Series(high_6h)
    low_6h_series = pd.Series(low_6h)
    close_6h_series = pd.Series(close_6h)
    
    highest_high = high_6h_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_6h_series.rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume 20-day average
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_ma = volume_ma_1d_aligned[i]
        vol = volume_1d_aligned[i]
        ema200 = ema_200_1w_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND volume > 2.0x avg AND price > 1w EMA200
            if wr < -80 and vol > 2.0 * vol_ma and price > ema200:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND volume > 2.0x avg AND price < 1w EMA200
            elif wr > -20 and vol > 2.0 * vol_ma and price < ema200:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R > -50 (return from oversold) OR volume spike ends
            if wr > -50 or vol <= 2.0 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R < -50 (return from overbought) OR volume spike ends
            if wr < -50 or vol <= 2.0 * vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_VolumeSpike_1wEMA200_Filter"
timeframe = "6h"
leverage = 1.0