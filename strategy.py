#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d ADX trend filter and volume confirmation.
# Long when Williams %R(14) < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period median.
# Short when Williams %R(14) > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period median.
# Williams %R identifies exhaustion points in trending markets; ADX filters for sufficient trend strength to avoid chop; volume confirms conviction.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.

name = "6h_WilliamsR_Extreme_1dADX_Volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Align Williams %R to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d ADX (14-period)
    plus_dm = np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])
    minus_dm = np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = np.maximum(
        np.maximum(
            np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])),
            np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
        ),
        np.abs(np.diff(df_1d['close'].values, prepend=df_1d['close'].values[0]))
    )
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_14 != 0, atr_14, 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_14 != 0, atr_14, 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / np.where((plus_di_14 + minus_di_14) != 0, (plus_di_14 + minus_di_14), 1e-10)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (wait for completed 1d bar)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Williams %R, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d ADX > 25 (sufficient trend strength)
        trending = adx_14_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND trending AND volume spike
            if williams_r_aligned[i] < -80 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought) AND trending AND volume spike
            elif williams_r_aligned[i] > -20 and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -50 (exit oversold) OR ADX < 20 (trend weakening)
            if williams_r_aligned[i] > -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -50 (exit overbought) OR ADX < 20 (trend weakening)
            if williams_r_aligned[i] < -50 or adx_14_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals