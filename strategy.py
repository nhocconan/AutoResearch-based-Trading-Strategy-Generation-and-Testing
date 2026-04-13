#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and chop regime filter.
    # Long when price breaks above Camarilla H3 + 1d volume > 1.5x 20-period average + CHOP(14) > 61.8.
    # Short when price breaks below Camarilla L3 + 1d volume > 1.5x 20-period average + CHOP(14) > 61.8.
    # Exit when price crosses Camarilla pivot point (PP).
    # Uses proven Camarilla structure with volume confirmation and range regime to avoid false breakouts.
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculations (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3
    # Range = High - Low
    range_1d = high_1d - low_1d
    # Resistance levels: R4 = Close + Range * 1.5/2, R3 = Close + Range * 1.25/2, etc.
    # Support levels: S4 = Close - Range * 1.5/2, S3 = Close - Range * 1.25/2, etc.
    # We use H3 (R3) and L3 (S3) for breakouts, PP for exit
    h3 = close_1d + (range_1d * 1.1 / 4)  # Camarilla R3
    l3 = close_1d - (range_1d * 1.1 / 4)  # Camarilla S3
    
    # Calculate True Range (TR) on 1d for Chopiness Index
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR(14) on 1d
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods on 1d
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index (CHOP) on 1d
    chop_denom = atr_1d * 14
    chop_num = hh_1d - ll_1d
    chop = np.where(chop_denom != 0, 100 * np.log10(chop_num / chop_denom) / np.log10(14), 50)
    
    # Get 1d data for volume spike (call ONCE before loop)
    volume_1d = df_1d['volume'].values
    
    # Calculate volume average (20-period) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirm = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for breakout fade in range)
        regime_filter = chop_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > h3_aligned[i]
        short_breakout = close[i] < l3_aligned[i]
        
        # Entry conditions: breakout + volume + regime
        long_signal = long_breakout and volume_confirm and regime_filter
        short_signal = short_breakout and volume_confirm and regime_filter
        
        # Exit conditions: price crosses Camarilla pivot point (PP)
        long_exit = close[i] < pp_aligned[i]
        short_exit = close[i] > pp_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_vol_chop_breakout_v1"
timeframe = "12h"
leverage = 1.0