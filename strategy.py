#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ADX trend filter + 1d Williams %R mean reversion + volume confirmation
# Long when ADX < 25 (weak trend/ranging) AND Williams %R < -80 (oversold) AND volume > 1.5x average
# Short when ADX < 25 AND Williams %R > -20 (overbought) AND volume > 1.5x average
# Exit when Williams %R returns to -50 level or ADX > 30 (strong trend)
# Targets 75-150 total trades over 4 years by combining trend filter with mean reversion in ranging markets
# Works in bull/bear markets: avoids trending regimes (ADX>30), fades extremes in ranging markets (ADX<25)

name = "12h_adx_williamsr_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX (14-period) from 12h data - trend strength filter
    # ADX < 25: ranging/weak trend (good for mean reversion)
    # ADX > 30: strong trend (avoid mean reversion)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], closed[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high - np.concatenate([[high[0]], high[:-1]])) > (np.concatenate([[low[0]], low[:-1]]) - low), 
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > (high - np.concatenate([[high[0]], high[:-1]])), 
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean()
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean()
    
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx = adx.values
    
    # Williams %R (14-period) from 1d timeframe for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - daily_close) / (highest_high - lowest_low + 1e-10)
    williams_r = williams_r.values
    
    # Align daily Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if required data not available
        if np.isnan(adx[i]) or np.isnan(williams_r_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: Williams %R returns to -50 level OR ADX indicates strong trend
        if position == 1:  # long position
            if williams_r_aligned[i] >= -50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if williams_r_aligned[i] <= -50 or adx[i] > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in weak trend/ranging market (ADX < 25) with Williams %R extremes
            # Long: Williams %R oversold (< -80) in ranging market + volume confirmation
            if (adx[i] < 25 and williams_r_aligned[i] < -80 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in ranging market + volume confirmation
            elif (adx[i] < 25 and williams_r_aligned[i] > -20 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals