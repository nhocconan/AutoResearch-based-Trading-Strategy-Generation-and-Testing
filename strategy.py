#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike filter and 12h chop regime.
    # Long when price breaks above Camarilla H3 + volume > 2.0x 20-period volume average + CHOP_12h > 61.8 (range).
    # Short when price breaks below Camarilla L3 + volume > 2.0x 20-period volume average + CHOP_12h > 61.8 (range).
    # Exit when price crosses Camarilla pivot point (PP).
    # Uses volume spike to confirm breakout strength and chop regime to avoid trend-following false breakouts.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day (need OHLC from 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # PP = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    PP_1d = (high_1d + low_1d + close_1d) / 3
    RANGE_1d = high_1d - low_1d
    R3_1d = close_1d + (RANGE_1d * 1.1 / 4)
    R4_1d = close_1d + (RANGE_1d * 1.1 / 2)
    S3_1d = close_1d - (RANGE_1d * 1.1 / 4)
    S4_1d = close_1d - (RANGE_1d * 1.1 / 2)
    
    # Align HTF Camarilla levels to 4h timeframe
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, R4_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, S4_1d)
    
    # Get 12h data for chop regime (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range (TR) on 12h
    tr1_12h = np.abs(high_12h - low_12h)
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]  # First period
    
    # Calculate ATR(14) on 12h
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods on 12h
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Calculate Chopiness Index (CHOP) on 12h
    chop_denom = atr_12h * 14
    chop_num = hh_12h - ll_12h
    chop_12h = np.where(chop_denom != 0, 100 * np.log10(chop_num / chop_denom) / np.log10(14), 50)
    
    # Align HTF chop to 4h timeframe
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    
    # Calculate 20-period volume average on 1d for volume spike filter
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(PP_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or np.isnan(R4_1d_aligned[i]) or
            np.isnan(S3_1d_aligned[i]) or np.isnan(S4_1d_aligned[i]) or np.isnan(chop_12h_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 20-period average
        volume_confirm = volume_1d[i] > 2.0 * vol_ma_20_1d[i]
        
        # Regime filter: CHOP_12h > 61.8 indicates ranging market (good for breakout fade)
        regime_filter = chop_12h_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > R3_1d_aligned[i]
        short_breakout = close[i] < S3_1d_aligned[i]
        
        # Entry conditions: breakout + volume + regime
        long_signal = long_breakout and volume_confirm and regime_filter
        short_signal = short_breakout and volume_confirm and regime_filter
        
        # Exit conditions: price crosses pivot point (PP)
        long_exit = close[i] < PP_1d_aligned[i]
        short_exit = close[i] > PP_1d_aligned[i]
        
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

name = "4h_1d_12h_camarilla_vol_chop_breakout_v1"
timeframe = "4h"
leverage = 1.0