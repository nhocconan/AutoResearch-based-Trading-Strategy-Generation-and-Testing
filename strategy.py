#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and 1w ADX trend filter.
# Long when price breaks above R1 with volume > 1.3x daily average and weekly ADX > 25 (trending market)
# Short when price breaks below S1 with volume > 1.3x daily average and weekly ADX > 25
# Exit when price returns to the pivot point (PP)
# Uses Camarilla for precise intraday levels, volume for conviction, ADX to avoid ranging markets.
# Target: 20-30 trades/year per symbol.

name = "4h_Camarilla_Volume_ADXTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # Using previous day's values (already closed)
    PP = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    R1 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.1 / 12
    S1 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP.values)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1.values)
    
    # Get 1d average volume for confirmation (20-period)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 1w ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate True Range and Directional Movement
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w.shift(1))
    tr3 = np.abs(low_1w - close_1w.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - high_1w.shift(1)
    down_move = low_1w.shift(1) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (14-period)
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # ADX (smoothed DX)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol = volume[i]
        pp = PP_aligned[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        adx_val = adx_aligned[i]
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_val > 25
        
        if position == 0:
            # Long entry: break above R1 + volume spike + trending market
            if price > r1 and vol > 1.3 * vol_ma and trending:
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 + volume spike + trending market
            elif price < s1 and vol > 1.3 * vol_ma and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point
            if price < pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point
            if price > pp:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals