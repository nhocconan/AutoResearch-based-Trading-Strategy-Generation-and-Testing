#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 1d ADX trend filter
# Reverses at key intraday support/resistance levels during trending markets with institutional volume
# Works in bull/bear by taking reversals at R1/S1 in uptrend and R3/S3 in downtrend
# Position size: 0.25 for balanced risk/return

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for volume, ADX, and pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ADX calculation (14-period)
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    
    # 1d Camarilla pivot levels (using previous day's OHLC)
    def calculate_camarilla(h, l, c):
        range_val = h - l
        return {
            'S1': c - range_val * 1.1 / 12,
            'S2': c - range_val * 1.1 / 6,
            'S3': c - range_val * 1.1 / 4,
            'R1': c + range_val * 1.1 / 12,
            'R2': c + range_val * 1.1 / 6,
            'R3': c + range_val * 1.1 / 4
        }
    
    # Calculate pivots for previous day
    camarilla_levels = calculate_camarilla(high_1d[-1], low_1d[-1], close_1d[-1])  # yesterday's levels
    camarilla_S1 = camarilla_levels['S1']
    camarilla_S3 = camarilla_levels['S3']
    camarilla_R1 = camarilla_levels['R1']
    camarilla_R3 = camarilla_levels['R3']
    
    # Align Camarilla levels to 4h (constant until new day)
    camarilla_S1_arr = np.full(len(close_1d), camarilla_S1)
    camarilla_S3_arr = np.full(len(close_1d), camarilla_S3)
    camarilla_R1_arr = np.full(len(close_1d), camarilla_R1)
    camarilla_R3_arr = np.full(len(close_1d), camarilla_R3)
    
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_arr)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_arr)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_arr)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_arr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ma20_1d_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_R1_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > volume_ma20_1d_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25
        
        # Combined filter
        filter_ok = volume_filter and trend_filter
        
        if position == 0:
            # Long reversal at S1 in uptrend OR S3 in strong downtrend (oversold bounce)
            if (close[i] <= camarilla_S1_aligned[i] and close[i-1] > camarilla_S1_aligned[i-1] and 
                adx_1d_aligned[i] > 25 and filter_ok):  # Uptrend reversal at S1
                signals[i] = 0.25
                position = 1
            elif (close[i] <= camarilla_S3_aligned[i] and close[i-1] > camarilla_S3_aligned[i-1] and 
                  adx_1d_aligned[i] > 30 and filter_ok):  # Strong oversold bounce at S3
                signals[i] = 0.25
                position = 1
            # Short reversal at R1 in downtrend OR R3 in strong uptrend (overbought pullback)
            elif (close[i] >= camarilla_R1_aligned[i] and close[i-1] < camarilla_R1_aligned[i-1] and 
                  adx_1d_aligned[i] > 25 and filter_ok):  # Downtrend reversal at R1
                signals[i] = -0.25
                position = -1
            elif (close[i] >= camarilla_R3_aligned[i] and close[i-1] < camarilla_R3_aligned[i-1] and 
                  adx_1d_aligned[i] > 30 and filter_ok):  # Strong overbought pullback at R3
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 (target) or R3 (extended target) or filter fails
            if (close[i] >= camarilla_R1_aligned[i] or 
                close[i] >= camarilla_R3_aligned[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 (target) or S3 (extended target) or filter fails
            if (close[i] <= camarilla_S1_aligned[i] or 
                close[i] <= camarilla_S3_aligned[i] or 
                not filter_ok):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Reversal_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0