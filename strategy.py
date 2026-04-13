#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 12h primary with 1d HTF - Camarilla pivot breakout with volume confirmation and chop regime filter
    # Designed to capture institutional breakouts from key daily pivot levels in both bull and bear markets
    # Target: 60-120 trades over 4 years (15-30/year) for low fee drag and good generalization
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla pivot levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # H3 = Close + 1.125*(High-Low), L3 = Close - 1.125*(High-Low)
    # We'll use H3/L3 for breakouts (more significant than H4/L4)
    cam_H3 = close_1d + 1.125 * (high_1d - low_1d)
    cam_L3 = close_1d - 1.125 * (high_1d - low_1d)
    cam_H4 = close_1d + 1.5 * (high_1d - low_1d)
    cam_L4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - np.roll(close, 1)[1:]))
        tr1 = np.maximum(tr1, np.abs(low[1:] - np.roll(close, 1)[1:]))
        tr = np.concatenate([[np.nan], tr1])
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(window)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_ma_10 = pd.Series(chop_1d).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    cam_H3_aligned = align_htf_to_ltf(prices, df_1d, cam_H3)
    cam_L3_aligned = align_htf_to_ltf(prices, df_1d, cam_L3)
    cam_H4_aligned = align_htf_to_ltf(prices, df_1d, cam_H4)
    cam_L4_aligned = align_htf_to_ltf(prices, df_1d, cam_L4)
    chop_ma_10_aligned = align_htf_to_ltf(prices, df_1d, chop_ma_10)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(cam_H3_aligned[i]) or 
            np.isnan(cam_L3_aligned[i]) or 
            np.isnan(cam_H4_aligned[i]) or
            np.isnan(cam_L4_aligned[i]) or
            np.isnan(chop_ma_10_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume_12h[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Chop regime filter: avoid extremely choppy markets (CHOP > 61.8) and strong trends (CHOP < 38.2)
        # We prefer moderate chop (38.2 <= CHOP <= 61.8) for mean reversion at pivot levels
        chop_regime = (chop_ma_10_aligned[i] >= 38.2) and (chop_ma_10_aligned[i] <= 61.8)
        
        # Breakout conditions at Camarilla H3/L3 levels
        breakout_up = close[i] > cam_H3_aligned[i]
        breakout_down = close[i] < cam_L3_aligned[i]
        
        # Entry conditions: breakout with volume confirmation in moderate chop regime
        enter_long = breakout_up and volume_confirmed and chop_regime
        enter_short = breakout_down and volume_confirmed and chop_regime
        
        # Exit conditions: price reaches opposite Camarilla level or returns to midpoint
        midpoint = (cam_H3_aligned[i] + cam_L3_aligned[i]) / 2
        exit_long = position == 1 and (close[i] <= midpoint or close[i] >= cam_H4_aligned[i])
        exit_short = position == -1 and (close[i] >= midpoint or close[i] <= cam_L4_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "12h_1d_camarilla_pivot_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0