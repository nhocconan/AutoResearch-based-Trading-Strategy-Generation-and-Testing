#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX + volume spike + choppiness regime filter
# TRIX (triple exponential average) identifies momentum with reduced lag.
# Volume spike confirms participation. Choppiness regime filter (CHOP > 61.8 = ranging) 
# enables mean reversion at Camarilla levels in ranging markets, trend following when trending.
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets via regime-adaptive logic.

name = "12h_TRIX_VolumeSpike_Chopper_Regime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d TRIX (15,9,9) for momentum
    close_1d = df_1d['close'].values
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = pd.Series(ema3).pct_change() * 100
    trix = trix_raw.values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_signal)
    
    # Calculate 1d CAMARILLA pivot levels (R1, S1, PP) for entry/exit
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Pivot Point (PP) = (H + L + C) / 3
    pp = (high_1d + low_1d + close_1d_arr) / 3.0
    # R1 and S1 levels
    r1 = pp + (high_1d - low_1d) * 1.1 / 12.0
    s1 = pp - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 1d CHOPPINESS INDEX (14) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * log(n))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d_arr[:-1]), 
                                np.abs(low_1d[1:] - close_1d_arr[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    atr1 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr1 / (14 * np.log10(14))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Get 12h volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20[i])
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # LONG ENTRY CONDITIONS
            long_signal = False
            if is_ranging:
                # In ranging market: mean reversion at S1
                long_signal = (close[i] <= s1_aligned[i] * 1.002 and volume_confirmed)
            elif is_trending:
                # In trending market: breakout above R1 with TRIX momentum
                long_signal = (close[i] > r1_aligned[i] and 
                               trix_aligned[i] > 0 and volume_confirmed)
            
            # SHORT ENTRY CONDITIONS
            short_signal = False
            if is_ranging:
                # In ranging market: mean reversion at R1
                short_signal = (close[i] >= r1_aligned[i] * 0.998 and volume_confirmed)
            elif is_trending:
                # In trending market: breakdown below S1 with TRIX momentum
                short_signal = (close[i] < s1_aligned[i] and 
                                trix_aligned[i] < 0 and volume_confirmed)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # EXIT LONG CONDITIONS
            exit_long = False
            if is_ranging:
                # Exit at PP or when price reaches R1 (take profit)
                exit_long = (close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i] * 0.995)
            elif is_trending:
                # Exit when TRIX turns negative or price breaks S1 (stop/reverse)
                exit_long = (trix_aligned[i] < 0 or close[i] < s1_aligned[i])
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # EXIT SHORT CONDITIONS
            exit_short = False
            if is_ranging:
                # Exit at PP or when price reaches S1 (take profit)
                exit_short = (close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i] * 1.005)
            elif is_trending:
                # Exit when TRIX turns positive or price breaks R1 (stop/reverse)
                exit_short = (trix_aligned[i] > 0 or close[i] > r1_aligned[i])
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals