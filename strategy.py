#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and 12h ADX trend filter
# Uses Camarilla levels from daily pivots for mean-reversion entries in ranging markets.
# Volume > 2x 20-bar median ensures institutional participation.
# 12h ADX > 25 filters for trending conditions to avoid false reversals in strong trends.
# Works in both bull (buy dips) and bear (sell rallies) markets by fading extremes.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    camarilla_h4 = pivot + 1.5 * range_
    camarilla_l4 = pivot - 1.5 * range_
    camarilla_h3 = pivot + 1.25 * range_
    camarilla_l3 = pivot - 1.25 * range_
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr2 = np.maximum(np.abs(low_12h[1:] - close_12h[:-1]), tr1)
    tr_12h = np.concatenate([[np.nan], tr2])
    
    # Calculate Directional Movement
    up_move = np.concatenate([[np.nan], high_12h[1:] - high_12h[:-1]])
    down_move = np.concatenate([[np.nan], low_12h[:-1] - low_12h[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_smooth = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price touches L3/L4 + volume spike + ADX < 25 (ranging market)
        if (close[i] <= camarilla_l3_aligned[i] and 
            volume[i] > vol_threshold[i] and 
            adx_12h_aligned[i] < 25):
            signals[i] = 0.25
        
        # Short: price touches H3/H4 + volume spike + ADX < 25 (ranging market)
        elif (close[i] >= camarilla_h3_aligned[i] and 
              volume[i] > vol_threshold[i] and 
              adx_12h_aligned[i] < 25):
            signals[i] = -0.25
        
        # Exit: price returns to pivot level
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] >= pivot[i // 16] if i // 16 < len(pivot) else pivot[-1]) or
               (signals[i-1] == -0.25 and close[i] <= pivot[i // 16] if i // 16 < len(pivot) else pivot[-1]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_CamarillaPivot_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0