#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme + 1d ADX Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20)
# with volume spike indicate potential reversals. 1d ADX > 25 ensures trades only in trending markets
# to avoid false signals in chop. Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Works in bull markets (buying oversold in uptrend) and bear markets (selling overbought in downtrend)
# by only taking trades in direction of 1d ADX trend (via +DI/-DI crossover).

name = "12h_WilliamsR_Extreme_1dADX_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d ADX (14-period) for trend strength
    # ADX requires +DI and -DI calculation
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = (pd.Series(high_1d) - pd.Series(low_1d).shift(1)).abs()
    tr4 = (pd.Series(low_1d) - pd.Series(high_1d).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1).values
    
    plus_dm = pd.Series(high_1d).diff()
    minus_dm = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_ma = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_di_ma = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * (plus_di_ma / tr_ma)
    minus_di = 100 * (minus_di_ma / tr_ma)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe (wait for completed 1d bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Determine trend direction from +DI/-DI (aligned)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    # Bullish trend: +DI > -DI, Bearish trend: -DI > +DI
    
    # Volume confirmation: 2.0x 20-period average (~10 days for 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Williams %R and ADX)
    start_idx = max(30, 40)  # 30 bars for Williams %R alignment, 40 for ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) with volume spike AND bullish trend (+DI > -DI)
            if (williams_r_aligned[i] < -80 and 
                volume_spike[i] and 
                plus_di_aligned[i] > minus_di_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought) with volume spike AND bearish trend (-DI > +DI)
            elif (williams_r_aligned[i] > -20 and 
                  volume_spike[i] and 
                  minus_di_aligned[i] > plus_di_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (return from oversold) OR trend changes to bearish
            if williams_r_aligned[i] > -50 or minus_di_aligned[i] > plus_di_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (return from overbought) OR trend changes to bullish
            if williams_r_aligned[i] < -50 or plus_di_aligned[i] > minus_di_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals